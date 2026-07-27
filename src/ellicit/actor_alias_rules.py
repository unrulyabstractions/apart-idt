"""Pairwise rules deciding when two normalized actor names name one actor.

``same_actor`` merges two names exactly when one of these rules fires:

1. Equal content tokens. Stopwords never distinguish actors, so "the united
   nations" and "united nations" are one name.
2. Acronym. One name is a single token of two or more letters spelling the
   initials of the other's content tokens: "un" and "united nations", "who"
   and "world health organization".
3. Token subset. One name's content tokens are a subset of the other's, and
   either the shorter name has at least two tokens ("doctors without borders"
   inside "doctors without borders medecins sans frontieres") or the two
   names differ by a single token ("macron" inside "emmanuel macron", the
   rule this module's predecessor applied). A bare one-token name therefore
   never crosses a gap of two or more words, so "france" stays out of
   "democratic party in france".
4. Reduced core. After dropping leading honorifics and one trailing generic
   institutional suffix plus its possessive remnant, one core's tokens are a
   subset of the other's, provided at least one of the two names was actually
   reduced: "mahatma gandhi" and "mohandas gandhi" via the honorific, and
   "macron administration", "macron s party", "emmanuel macron" via the
   suffix. A name that lost nothing gains no new merges from this rule.

The honorific and suffix vocabularies are small explicit lists, never fuzzy
matches. The rules are deliberately one-actor-conservative: two names that
share no tokens never merge ("emmanuel macron" and "marine le pen"), and the
caller (``consolidate_variants``) additionally refuses any merge whose name
matches two or more distinct canonical names, so a bare surname facing both
"mohandas gandhi" and "indira gandhi" stays separate rather than guessing.
"""

from __future__ import annotations

__all__ = ["acronym_match", "content_tokens", "core_tokens", "same_actor"]

#: Dropped before comparing actor names, so "the democratic party" and
#: "democratic party" are one candidate rather than two.
_STOPWORDS = frozenset({"the", "a", "an", "of", "in", "for"})

#: Leading titles that are not part of who the actor is. Small and explicit;
#: growing it needs a measured duplicate, not a hunch.
_HONORIFICS = frozenset({"mahatma", "mr", "mister", "mrs", "ms", "miss", "dr",
                         "doctor", "sir", "saint", "st", "pope", "president",
                         "senator"})

#: A trailing one of these turns a person into a derived institution; the
#: tally wants the person. Small and explicit, same bar as honorifics.
_INSTITUTIONAL_SUFFIXES = frozenset({"administration", "party", "government",
                                     "campaign", "foundation", "team"})


def content_tokens(actor: str) -> tuple[str, ...]:
    """The name's tokens with stopwords dropped, in order."""
    return tuple(t for t in actor.split() if t not in _STOPWORDS)


def core_tokens(actor: str) -> tuple[tuple[str, ...], bool]:
    """(content tokens minus honorifics and one institutional suffix, was_reduced).

    "macron s party" reduces to ("macron",): the suffix goes first, then the
    possessive remnant "s" it exposes. "mahatma gandhi" reduces to ("gandhi",).
    """
    tokens = list(content_tokens(actor))
    original = len(tokens)
    while tokens and tokens[0] in _HONORIFICS:
        tokens.pop(0)
    if tokens and tokens[-1] in _INSTITUTIONAL_SUFFIXES:
        tokens.pop()
        if tokens and tokens[-1] == "s":
            tokens.pop()
    return tuple(tokens), len(tokens) < original


def acronym_match(a: str, b: str) -> bool:
    """True when one name is the initials of the other's content tokens."""
    for short, long in ((a, b), (b, a)):
        tokens = short.split()
        spelled = content_tokens(long)
        if (len(tokens) == 1 and len(tokens[0]) >= 2 and tokens[0].isalpha()
                and len(spelled) >= 2
                and tokens[0] == "".join(t[0] for t in spelled)):
            return True
    return False


def same_actor(a: str, b: str) -> bool:
    """Whether two normalized names name one actor, per the module rules."""
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return False
    sa, sb = frozenset(ta), frozenset(tb)
    if sa == sb:
        return True
    if acronym_match(a, b):
        return True
    small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    if small < big and (len(small) >= 2 or len(big) - len(small) <= 1):
        return True
    (ca, reduced_a), (cb, reduced_b) = core_tokens(a), core_tokens(b)
    if not (reduced_a or reduced_b) or not ca or not cb:
        return False
    csa, csb = frozenset(ca), frozenset(cb)
    return csa <= csb or csb <= csa
