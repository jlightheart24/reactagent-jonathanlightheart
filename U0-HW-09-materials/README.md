# U0-HW-09 Source Documents

These are the complete source documents distributed with the assignment. The
student-facing query set is `U0-HW-09-required-queries.md` in this directory.

`model-service.env.example` provides the course embedding and chat service
settings. Copy the settings your project needs into a local `.env`, replace the
placeholder API keys with your assigned credential, and connect to the course
network before using either service. Never commit `.env` or credentials. The
classroom endpoints use HTTP and send the bearer credential without transport
encryption, so use them only on the institution-approved course network.

Retrieved from the official sources on October 21, 2026.

| File | Source | Pages | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `documents/fda-food-code-2026.pdf` | [FDA Food Code 2026](https://www.fda.gov/media/194741/download?attachment) | 732 | 5,215,431 | `a608605db1a44d07fc9255db281175a933b4290597b4dc4cf58e035158fc8e06` |
| `documents/fda-employee-health-personal-hygiene-handbook.pdf` | [Employee Health and Personal Hygiene Handbook](https://www.fda.gov/media/77065/download?attachment) | 45 | 3,144,073 | `d83321e689eab26958f27ee5dd8886d231b9140fdc9cbcca3e30613da80c74e8` |
| `documents/fda-bad-bug-book-second-edition.pdf` | [Bad Bug Book, Second Edition](https://www.fda.gov/media/83271/download) | 292 | 5,252,559 | `b76d65edc9cce029a8e864f51eee8f7ae124157d55532019c6ea9de529ff21d7` |
| `documents/fda-appendix-3-bacterial-pathogen-growth-inactivation.pdf` | [Appendix 3: Bacterial Pathogen Growth and Inactivation](https://www.fda.gov/media/99598/download?attachment) | 23 | 190,124 | `ba818268cc7d6894f41c0200f14b064ad254da8f649cf7f75d62ca03733207c1` |
| `documents/utah-risk-factor-study-2016-2017.pdf` | [Utah Risk Factor Study, 2016-2017](https://epi.utah.gov/wp-content/uploads/Utah-Risk-Factor-Study_2016-2017.pdf) | 18 | 533,573 | `ea1f93b84c18e73d9ffa2f7909705ea0012eef08a440384254e4f5bc7c516737` |

**Total:** 1,110 pages and 14,335,760 bytes.

The complete corpus is intentional. Its size and uneven source distribution
require focused queries, source filtering, and evidence checks while remaining
small enough to distribute and index on student hardware. Do not replace the
documents with excerpts.

## Source Roles

- The FDA Food Code is a model code. Jurisdiction-specific adoption requires a
  separate authority.
- The employee-health handbook reflects an earlier Food Code edition and serves
  as operational guidance.
- The *Bad Bug Book* provides pathogen and toxin reference information.
- Appendix 3 is draft, nonbinding scientific guidance for food processing.
- The Utah study reports historical inspection observations. Current legal
  requirements require a separate authority.
