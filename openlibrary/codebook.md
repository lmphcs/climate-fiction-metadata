# CROW metadata annotation codebook v1.0

Status: **frozen for held-out evaluation**
Unit: one Open Library work-level metadata record

All judgments mean **"as represented in the supplied metadata."** Use only the
displayed title, subtitle, authors, year, subjects, and description. Do not use
prior knowledge, external lookup, the Open Library URL, retrieval-route
information, or model scores. If prior knowledge is unavoidable, record it in
the optional researcher comment; do not silently use it as evidence.

Version 1.0 incorporates the completed development calibration. Sparse
bibliographic metadata often does not support a defensible causal distinction
between environmental framing and implicit climate framing. The primary target
is therefore **environment-or-climate relevance as represented in supplied
metadata**. Direct climate language remains a distinct explicit subgroup.

## 1. Decision order

1. Assign fiction_status.
2. If NONFICTION or OTHER_LITERARY, stop primary coding.
3. If FICTION or UNCLEAR, assign one environment_climate_relevance label.
4. Optionally enter a short evidence span for a positive or unclear decision.
5. Always offer one optional annotation_comment, including after an early
   exit. It may contain decision reasoning or reflexive notes, but it is not
   primary label evidence.

## 2. Fiction status

- FICTION: metadata explicitly identifies a novel, novella, short story, story
  collection, prose-fiction anthology, or prose fiction.
- NONFICTION: metadata explicitly identifies memoir, biography, report, guide,
  textbook, scholarship, journalism, scholarly/argumentative essays, or
  another factual or instructional form.
- OTHER_LITERARY: poetry, drama, graphic narrative, or another literary form
  outside this pilot's prose-fiction target.
- UNCLEAR: work type is absent, generic, mixed without a clear prose-fiction
  component, or contradictory.

Collections are classified by represented form, not number of authors. A
collection or anthology explicitly presented as short stories/prose fiction is
FICTION. A collection of scholarly or argumentative essays is NONFICTION. For
a mixed anthology, use FICTION only when metadata explicitly establishes a
substantive prose-fiction component; otherwise use UNCLEAR and note ambiguity.

## 3. Environment-or-climate relevance

- NONE: no environmental or climate framing is established, or a retrieved
  cue is incidental, metaphorical, or non-environmental.
- IMPLICIT_OR_ENVIRONMENTAL: supplied metadata establishes environmental,
  ecological, species, resource, energy-transition, pollution, natural-hazard,
  or non-explicit climate framing, but does not directly name climate change or
  an equivalent formulation.
- CLIMATE_EXPLICIT: supplied metadata explicitly names climate change, the
  climate crisis, global warming, climatic change, anthropogenic warming, or an
  equivalent direct climate formulation.
- UNCLEAR: supplied metadata is too sparse, contradictory, or genuinely
  ambiguous to choose among the other labels.

IMPLICIT_OR_ENVIRONMENTAL does not assert climate causation. An isolated
drought, flood, wildfire, storm, heatwave, extinction, or pollution event may
support it only when the supplied metadata presents that condition as an
environmental framing of the work rather than incidental scenery. A mine,
pipeline, oil company, apocalypse, ordinary weather event, or environmental
word used metaphorically is not automatically relevant. Code NONE when no
environment-or-climate framing is established; do not supply unstated plot or
causal links.

## 4. Collections and component-level discoverability

The unit remains the work, but discovery may be triggered by a represented
component. If metadata explicitly identifies a climate-related story or other
substantive prose-fiction component inside a wider fiction collection, code the
work using the climate label supported for that component. A comment may note
"component-level framing."

This means the work is discoverable through its supplied paratext. It does not
establish that climate organizes the whole collection or every component. Do
not infer the content of unmentioned stories.

## 5. Evidence and annotation comment

- evidence_span: optional short phrase that helped the decision. Exact copying
  is preferred when effortless, but this field is non-gold and may be blank.
- evidence_field: inferred by the utility; title, subtitle, subjects,
  description, multiple, not_found, or blank are possible. not_found is a
  process warning, not proof that the label is wrong.
- annotation_comment: optional combined field for concise decision reasoning,
  prior knowledge, affect, reading interest, interface feedback, or
  methodological reflection. It is not evidence for the primary judgment.

Do not infer symbolism, scientific accuracy, stance, author beliefs, reader
beliefs, population prevalence, or unmentioned plot events.

## 6. Derived target status

- Positive: FICTION plus IMPLICIT_OR_ENVIRONMENTAL or CLIMATE_EXPLICIT.
- Negative: NONFICTION, OTHER_LITERARY, or FICTION plus NONE.
- Unresolved: UNCLEAR work type, UNCLEAR climate relevance, or an incomplete
  eligible record.

Explicit and non-explicit positives remain separate for analysis. Annotators
do not enter the derived status.

## 7. Synthetic examples

| Metadata excerpt | Codes | Reason |
|---|---|---|
| "A novel about families displaced by anthropogenic climate change." | FICTION, CLIMATE_EXPLICIT | Climate change is directly named. |
| "Decades of rising oceans, lethal summers, and repeated crop failures reshape an island society." | FICTION, IMPLICIT_OR_ENVIRONMENTAL | A systemic non-explicit climate/environmental frame is represented. |
| "A detective investigates ecological damage during a severe drought." | FICTION, IMPLICIT_OR_ENVIRONMENTAL | Environmental framing is represented; no climate-causation claim follows. |
| "A thriller about corruption at an oil company." | FICTION, NONE | An energy-industry setting alone is not necessarily environmental or climatic. |
| "Stories of wildlife conservation and habitat loss." | FICTION, IMPLICIT_OR_ENVIRONMENTAL | Ecological framing is explicit without asserting a climate link. |
| "A story collection including 'After the Climate Floods.'" | FICTION, CLIMATE_EXPLICIT | Component-level paratext makes the work discoverable; no whole-collection centrality claim follows. |
| "Scholarly essays on climate narratives." | NONFICTION; stop | The represented form is scholarship. |
| Title: Heat; no usable form or topic metadata | UNCLEAR, UNCLEAR | Neither construct is established. |

## 8. Freeze and change control

This codebook and its derived-status mapping are frozen before held-out
annotation. Held-out judgments must not be silently reinterpreted. Any later
construct change requires a new version, an explicit sensitivity analysis, and
preservation of the original v1.0 labels.
