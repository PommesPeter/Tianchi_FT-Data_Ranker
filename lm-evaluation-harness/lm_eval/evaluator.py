import collections
import itertools
import random
import json
import os
import pathlib

import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.utils import positional_deprecated, run_task_tests
from lm_eval.models.gpt2 import HFLM

import numpy as np
import transformers


@positional_deprecated
def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    max_batch_size=None,
    device=None,
    no_cache=False,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    check_integrity=False,
    decontamination_ngrams_path=None,
    detail_output_path=None,
    infer_only=False,
    eval_only=False,
    concise_output=False
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model, transformers.PreTrainedModel object, or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param detail_output_path: str, optional
        Path to which detailed eval info will be written.
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

    if not eval_only:
        if isinstance(model, str):
            if model_args is None:
                model_args = ""
            lm = lm_eval.models.get_model(model).create_from_arg_string(
                model_args, {"batch_size": batch_size, "max_batch_size": max_batch_size, "device": device}
            )
        elif isinstance(model, transformers.PreTrainedModel):
            lm = lm_eval.models.get_model("hf-causal")(
                    pretrained=model,
                    batch_size=batch_size,
                    max_batch_size=max_batch_size,
                    )
            no_cache = True
        else:
            assert isinstance(model, lm_eval.base.LM)
            lm = model

        if not no_cache:
            lm = lm_eval.base.CachingLM(
                lm,
                "lm_cache/"
                + (model if isinstance(model, str) else model.model.config._name_or_path)
                + "_"
                + model_args.replace("=", "-").replace(",", "_").replace("/", "-")
                + ".db",
            )
    else:
        lm = None

    task_dict = lm_eval.tasks.get_task_dict(tasks)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
        detail_output_path=detail_output_path,
        infer_only=infer_only,
        eval_only=eval_only,
        concise_output=concise_output
    )

    # add info about the model and few shot config
    model_name = None
    if isinstance(model, str):
        model_name = model
    elif isinstance(model, transformers.PreTrainedModel):
        model_name = "pretrained=" + model.config._name_or_path
    results["config"] = {
        "model": model_name,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "batch_sizes": list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else [],
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
        "description_dict": description_dict,
    }

    return results


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    provide_description=None,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
    detail_output_path=None,
    infer_only=False,
    eval_only=False,
    concise_output=False
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param detail_output_path: str, optional
        Path to which detailed eval info will be written.
    :return
        Dictionary of results
    """
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    # TODO: todo: implement proper description-providing system
    assert not provide_description  # not implemented.
    if provide_description is not None:
        # nudge people to not specify it at all
        print(
            "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
        )

    decontaminate = decontamination_ngrams_path is not None

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}
    write_detail_info = {}
    docs_for_decontamination = collections.defaultdict(list)

    write_detail = detail_output_path is not None and not eval_only

    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        # print(f"Task: {task_name}; number of docs: {len(task_docs)}")

        if write_detail:
            prompt_details = []

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )

        if limit is not None:
            limit_num = int(len(task_docs) * limit) if limit < 1.0 else int(limit)
            if 0 < limit_num < len(task_docs):
                rnd.shuffle(task_docs)
                rnd.seed(42)
        else:
            limit_num = None

        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit_num)):
            if decontaminate and task.should_decontaminate():
                docs_for_decontamination[(task_name, task_set)].append(
                    task.doc_to_decontamination_query(doc)
                )

            docs[(task_name, doc_id)] = doc
            ctx = task.fewshot_context(
                doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
            )
            reqs = task.construct_requests(doc, ctx)

            if write_detail:
                prompt_details.append({"task_name": task_name, "doc_id": doc_id, 'num_reqs': len(reqs)})

            # print the prompt for the first few documents
            # if doc_id < 1 and not eval_only:
            #     print(
            #         f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
            #     )
            #     print("Requests:", reqs)

            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            if not eval_only:
                for i, req in enumerate(reqs):
                    requests[req.request_type].append(req)
                    # i: index in requests for a single task instance
                    # doc_id: unique id that we can get back to a doc using `docs`
                    requests_origin[req.request_type].append((i, task_name, doc, doc_id))

                    if write_detail and not concise_output:
                        prompt_details[-1][f"prompt_{i}"] = "".join(
                            (map(lambda x: "".join(x), req.args))
                        )

        if write_detail:
            write_detail_info[task_name] = prompt_details

    # Compare all tasks/sets at once to ensure a single training set scan
    if decontaminate:
        from lm_eval.decontamination.decontaminate import get_train_overlap

        print("Finding train/test overlap, please wait...")
        overlaps = get_train_overlap(
            docs_for_decontamination, decontamination_ngrams_path, limit_num
        )

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    # execute each type of request
    if eval_only:
        print(f"Loading inference results of task {task_dict.keys()}")
        try:
            inference_results = load_detail_output(detail_output_path, task_dict.keys())
        except Exception as e:
            print("Fail to load inference results, please check --detail_output_path")
            exit(1)
        for result in inference_results:
            for i in range(result['num_reqs']):
                process_res_queue[(result['task_name'], result['doc_id'])].append((i, result[f'logit_{i}']))
    else:
        for reqtype, reqs in requests.items():
            # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
            #       only in index. We could implement some kind of caching, but that would be more of a band-aid
            #       solution. we could also implement some kind of auto-grouping here;
            #       they should end up next to each other.

            print("Running", reqtype, "requests")
            resps = getattr(lm, reqtype)([req.args for req in reqs])
            resps = [
                x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
            ]

            for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
                process_res_queue[(task_name, doc_id)].append((i, resp))

                if write_detail:
                    write_detail_info[task_name][doc_id][f"logit_{i}"] = resp
                    task = task_dict[task_name]
                    if not infer_only:
                        if isinstance(task, lm_eval.base.MultipleChoiceTask):
                            write_detail_info[task_name][doc_id]["truth"] = doc["gold"]
                        elif isinstance(task, lm_eval.tasks.winogrande.Winogrande):
                            write_detail_info[task_name][doc_id]["truth"] = task.answer_to_num[
                                doc["answer"]
                            ]
                        else:
                            write_detail_info[task_name][doc_id]["truth"] = task.doc_to_target(doc)

    if not infer_only:
        vals = collections.defaultdict(list)

        # unpack results and sort back in order and return control to Task
        for (task_name, doc_id), requests in process_res_queue.items():
            requests.sort(key=lambda x: x[0])
            requests = [x[1] for x in requests]

            task = task_dict[task_name]
            doc = docs[(task_name, doc_id)]

            metrics = task.process_results(doc, requests)
            for metric, value in metrics.items():
                vals[(task_name, metric)].append(value)

                if write_detail:
                    write_detail_info[task_name][doc_id][metric] = str(value)

                # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
                if decontaminate and task_name in overlaps:
                    if doc_id not in overlaps[task_name]:
                        vals[(task_name, metric + decontaminate_suffix)].append(value)

        # aggregate results
        for (task_name, metric), items in vals.items():
            task = task_dict[task_name]
            real_metric = metric  # key when looking up the metric with task.aggregation
            if metric.endswith(decontaminate_suffix):
                real_metric = metric.replace(
                    decontaminate_suffix, ""
                )  # decontaminated still uses the same metric
            results[task_name][metric] = task.aggregation()[real_metric](items)

            # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
            # so we run them less iterations. still looking for a cleaner way to do this

            stderr = lm_eval.metrics.stderr_for_metric(
                metric=task.aggregation()[real_metric],
                bootstrap_iters=min(bootstrap_iters, 1000)
                if metric in ["bleu", "chrf", "ter"]
                else bootstrap_iters,
            )

            if stderr is not None:
                results[task_name][metric + "_stderr"] = stderr(items)

    if write_detail:
        detail_output_path = (
            pathlib.Path(detail_output_path)
            if detail_output_path is not None
            else pathlib.Path(".")
        )
        try:
            detail_output_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        for task_name, _ in task_dict_items:
            with open(
                detail_output_path.joinpath(f"{task_name}_detail.json"),
                "w",
                encoding="utf8",
            ) as fp:
                json.dump(write_detail_info[task_name], fp, indent=4, ensure_ascii=False)

    return {"results": dict(results), "versions": dict(versions)}


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()

def load_detail_output(detail_output_path, task_names):
    eval_results = []
    for task_name in task_names:
        file_path = os.path.join(detail_output_path, f"{task_name}_detail.json")
        print(f"load from {file_path}")
        with open(file_path, "r", encoding='utf8') as file:
            json_data = json.load(file)
            eval_results.extend(json_data)
    return eval_results