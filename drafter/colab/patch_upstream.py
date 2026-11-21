from pathlib import Path

import sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state as aux_state
import specforge
import specforge_sglang_capture.capture as capture

SPECFORGE = Path(specforge.__file__).parent

DEFER_ON_RELEASE = '''

def _defer_on_release(scheduler):
    pending = {}
    retirer = scheduler.request_retirer
    previous = retirer.release_observer

    def observe(req):
        if previous is not None:
            previous(req)
        completion = pending.pop(req.rid, None)
        if completion is not None:
            scheduler.defer_response(req, completion)

    retirer.release_observer = observe

    def defer(req, completion):
        if req.owner_list is None:
            scheduler.defer_response(req, completion)
        else:
            pending[req.rid] = completion

    return defer


def install(scheduler) -> CaptureRuntime:'''

LOOP_VISITS_OLD = '''        count = target.get("num_hidden_layers")
        positive_int("target_config.num_hidden_layers", count)
        if any(layer >= count for layer in self.target_layer_ids):'''

LOOP_VISITS_NEW = '''        count = target.get("num_hidden_layers")
        positive_int("target_config.num_hidden_layers", count)
        loop = target.get("loop_config")
        if loop:
            visits = count + (loop["num_passes"] - 1) * (loop["core_end"] - loop["core_start"])
            if any(layer > visits for layer in self.target_layer_ids) or self.target_layer_ids[-1] != visits:
                raise ValueError("loop targets tap layer visits and end with the post-norm visit count")
            return
        if any(layer >= count for layer in self.target_layer_ids):'''

PATCHES = [
    (
        "sglang: a loop target's post-norm tap equals the visit count",
        Path(aux_state.__file__),
        "if layer_id >= num_layers]",
        "if layer_id > num_layers]",
    ),
    (
        "capture plugin: DisaggregationMode moved to sglang.srt.disaggregation.mode",
        Path(capture.__file__),
        "from sglang.srt.disaggregation.utils import DisaggregationMode",
        "from sglang.srt.disaggregation.mode import DisaggregationMode",
    ),
    (
        "capture plugin: defer a capture response when SGLang releases it",
        Path(capture.__file__),
        "\n\ndef install(scheduler) -> CaptureRuntime:",
        DEFER_ON_RELEASE,
    ),
    (
        "capture plugin: route deferral through the release hook",
        Path(capture.__file__),
        "defer_response=scheduler.defer_response,",
        "defer_response=_defer_on_release(scheduler),",
    ),
    (
        "specforge: loop targets tap layer visits",
        SPECFORGE / "config/draft/architecture/features.py",
        LOOP_VISITS_OLD,
        LOOP_VISITS_NEW,
    ),
    (
        "specforge: greedy drafting treats the monotone final softcap as neutral",
        SPECFORGE / "config/draft/codec/target.py",
        '        "final_logit_softcapping": None,\n',
        "",
    ),
]


def apply(name, path, old, new):
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new, 1))
        return "applied"
    if new in text:
        return "already applied"
    raise SystemExit(f"patch_upstream: {name}: anchor not found in {path}")


for name, path, old, new in PATCHES:
    print(f"{apply(name, path, old, new)}: {name}")