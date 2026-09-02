+++
id = "ADR-0001"
title = "Reject implausibly compressed word timelines"
status = "accepted"
invariant = "Provider and caller word timelines above 4.5 words per speaking second are not trusted for speaker attribution."
on_change = [
  "Re-evaluate the threshold against representative Swedish speech recordings.",
  "Verify long pauses cannot hide a locally compressed timeline.",
  "Run the word and segment plausibility tests and inspect speaker attribution outcomes.",
]
evidence = "README.md#word-timestamps-trust-and-the-alignment-rungs"
+++

# Context

Decoder-derived timestamps have produced compressed timelines that place
speaker turns on the wrong words. Vemsa therefore rejects implausible word and
segment windows and force-aligns the transcript against the audio. The current
4.5 words-per-speaking-second threshold is a reviewed quality boundary, not a
general performance tuning value.
