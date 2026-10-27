# Claim-support review of the required runs

All nine required queries were run once with the configuration in [`config.json`](config.json), and none was
rerun. Citation matching confirmed that every cited `chunk_id` was returned by retrieval during its run; the
controller enforces this. This review covers a separate question: does each claim follow from its cited passage?
I checked claims against the stored chunk text and gave the most attention to numbers, table values, and
section attributions.

## Outcome summary

| Query | Expected | Result | Steps | Claim review |
|---|---|---|---:|---|
| Q1 | answer | answer | 6 | Supported, except the final claim's explanation of "P" (see below) |
| Q2 | answer | answer | 5 | Supported; source-year metadata note |
| Q3 | answer | answer | 6 | Supported |
| Q4 | answer | answer | 2 | Supported; all 14 Table 3-A values verified |
| Q5 | answer | answer | 6 | Supported |
| Q6 | answer | answer | 8 | Supported; section-label note |
| Q7 | answer | answer | 8 | Supported; section-label note |
| Q8 | insufficient_evidence | insufficient_evidence | 5 | Correct: no Utah adoption rule or penalties in the corpus |
| Q9 | insufficient_evidence | insufficient_evidence | 2 | Correct: off-topic; one search returned unrelated passages |

Every query ended with its expected terminal reason.

## Findings

### Unsupported claim (Q1, final claim)

The final claim says the "P" marker on § 2-201.12 means a "provision addressed by a core or noncore item." The
cited chunk `food-code-2026:p59:c0` shows the letter "P" but never defines it. In the Food Code, "P" marks a
*Priority* item. The explanation therefore came from the model's own knowledge, not from retrieved evidence, and
it is wrong. The other five Q1 claims, which state the exclusion rule and the 24-hour or medical-documentation
reinstatement rule, are directly supported by `p59:c0`, `p61:c0`, `p388:c0`, `p404:c0`, `p391:c1`, and `p63:c1`.

### Facts taken from source-role metadata (Q2, Q3, Q6, Q7)

The automatic numeric check flagged years such as "2012", "2020", "2017", and "2026" in claims that describe a
source's status. These years come from the `source_role` label that the application attaches to every retrieval
result, not from the passage text, so they are application metadata and not fabrications. Q7's handbook-edition
claim was checked against the passage itself: `employee-health-handbook:p3:c0` ("updated to reflect … the 2017
FDA Food Code") and `p4:c2` ("Adapted from the 2017 FDA Food Code").

### Section labels from neighboring text (Q6, Q7)

Some claims cite ¶ 2-201.11(A)(1)(a) for "vomiting is a reportable symptom." The cited chunk
`food-code-2026:p56:c1` contains the list "Reportable symptoms (1) … (a) Vomiting," but the section heading
"2-201.11" is in the preceding chunk. The substance is supported; the section number is inferred from document
structure. This is a known effect of chunking (see the report's limitations).

## Verified high-risk details

- **Q3:** The infective dose is "as low as 1 to 10 viral particles." Onset is "between 24 and 48 hours (median
  in outbreaks: 33 to 36 hours) … within 12 hours … reported" (`bad-bug-book:p169:c0`).
- **Q4:** Table 3-A on `appendix-3:p3:c0`:

  | Pathogen | Min. aw | pH | Max. salt | Temperature | Oxygen |
  |---|---|---|---|---|---|
  | *Salmonella* spp. | 0.94 | 3.7–9.5 | 8% | 41.4–115.2 °F | facultative anaerobe |
  | *L. monocytogenes* | 0.92 | 4.4–9.4 | 10% | 31.3–113 °F | facultative anaerobe |

  These match the agent's claims. The rows are not shifted, even though PDF extraction flattens the table.
- **Q5:** The four line items and their rate ratios match `utah-risk-factor-study:p4:c1`: 22 at 1.37, 28 at 1.15,
  33 at 1.50, and 36 at 1.23. The limitations match `p4:c2`, `p5:c1`, `p5:c2`, and `p6:c3`.
- **Q2:** "Stop work immediately; Report to management; and Go home and return after at least 24 hours …"
  (`employee-health-handbook:p6:c0`). The manager actions match `p7:c2`.
- **Q6:** A nursing home is named in the highly-susceptible-population definition (`food-code-2026:p38:c0`). The
  answer separates the Bad Bug Book's dose (1 to 10 particles) from the Food Code Annex figure (as few as 18
  particles, `food-code-2026:p374:c0`) and attributes each to its source. It treats vomiting as a symptom and
  gives norovirus-specific rules only as a conditional ("if norovirus were subsequently diagnosed").
