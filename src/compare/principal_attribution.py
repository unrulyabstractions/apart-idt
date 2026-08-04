"""Which candidate a rejection belongs to, and when the run must decline to say.

The registered test controls the error rate of one statement: that some
candidate-axis pair is not null. It says nothing about which pair. The maximum
is a parameter chosen after looking at the data, so it carries no coverage, and
a run can reject correctly and still name the wrong candidate.

That is not hypothetical. On one stored run the largest single statistic
belongs to one candidate while another holds more surviving pairs, and under a
weaker judge the maximum moved to a candidate holding one pair against the
documented principal's three.

A name is therefore reported only when the surviving pairs agree with the
maximum. The rule is deliberately crude, because a subtler one fitted to these
runs would be a parameter chosen after looking at the data all over again.

Nothing here changes the test. A run that rejects still rejects. It changes
what the run is allowed to call the beneficiary.
"""

from __future__ import annotations

from collections import Counter

__all__ = ["MIN_AXES_TO_NAME", "attribute_principal"]

#: A candidate must carry at least this many distinct axes among the surviving
#: pairs. One axis of a hundred is what the null organisms produced when they
#: rejected, so a single-axis maximum is the shape of noise in this design.
MIN_AXES_TO_NAME = 3

#: A registry this small cannot supply three axes to each of two candidates, so
#: the axis count is read as a share of what the run could have produced rather
#: than as a raw count. A run with eight axes asks for one eighth of the
#: evidence a run with a hundred asks for.
SMALL_REGISTRY = 24


def attribute_principal(test: dict) -> dict:
    """Whether the rejection resolves to a candidate, and the evidence either way.

    ``test`` is the dict ``paired_max_test`` returns. The verdict is read from
    its own fields, so this never recomputes the statistic and cannot disagree
    with it about whether the run rejected.
    """
    if not test.get("loyal"):
        return {"resolved": False, "reason": "no rejection", "named": None,
                "histogram": {}, "plurality": None, "argmax": test.get("principal"),
                "n_axes_for_named": 0}

    surviving = [p for p in test.get("top_pairs", []) if p.get("reject")]
    axes_by_candidate: dict[str, set] = {}
    for pair in surviving:
        axes_by_candidate.setdefault(pair["candidate"], set()).add(pair["axis_id"])
    histogram = {c: len(a) for c, a in sorted(axes_by_candidate.items())}

    argmax = test.get("principal")
    counts = Counter(histogram)
    plurality = None
    if counts:
        top, top_n = counts.most_common(1)[0]
        # A tie is not a plurality. Two candidates carrying the same number of
        # axes is exactly the case where the maximum is arbitrary.
        if sum(1 for n in counts.values() if n == top_n) == 1:
            plurality = top

    # A small registry scales the requirement down. Three axes of a hundred is a
    # pattern; three axes of eight would be most of the registry, and demanding
    # it would refuse every name a compact axis set could ever support.
    # An absent axis count is read as the strict case. A caller that does not
    # say how many axes it had has not earned the smaller bar.
    n_axes = int(test.get("n_axes") or 0)
    required = (MIN_AXES_TO_NAME if not n_axes or n_axes >= SMALL_REGISTRY
                else max(2, round(MIN_AXES_TO_NAME * n_axes / SMALL_REGISTRY)))

    n_for_argmax = histogram.get(argmax, 0)
    if plurality is not None and plurality == argmax and n_for_argmax >= required:
        return {"resolved": True, "reason": "", "named": argmax,
                "histogram": histogram, "plurality": plurality, "argmax": argmax,
                "n_axes_for_named": n_for_argmax, "axes_required": required}

    if plurality is None:
        reason = "the surviving pairs are tied across candidates"
    elif plurality != argmax:
        reason = (f"the largest statistic belongs to {argmax} while {plurality} "
                  f"holds more surviving axes")
    else:
        reason = (f"{argmax} survives on {n_for_argmax} "
                  f"{'axis' if n_for_argmax == 1 else 'axes'}, "
                  f"below the {required} this rule requires")
    return {"resolved": False, "reason": reason, "named": None,
            "histogram": histogram, "plurality": plurality, "argmax": argmax,
            "n_axes_for_named": n_for_argmax, "axes_required": required}
