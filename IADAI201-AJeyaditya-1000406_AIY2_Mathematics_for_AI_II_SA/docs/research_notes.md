# Research notes - Hospital Inpatient Discharges

## How hospital administrators use discharge data
| Theme | What they look at | Why |
| --- | --- | --- |
| Capacity | Length of stay (mean, median, 90th/95th percentile), bed-days | Beds and nursing hours are the scarcest resources; a one-day reduction frees capacity for the next patient |
| Cost | Total charges, costs, cost-to-charge ratio, charge per day | Budgeting, pricing and payer negotiation |
| Case mix | APR-DRG severity of illness, age group, diagnosis group | Fair comparison between hospitals/departments - raw averages are misleading without risk adjustment |
| Payer mix | Share of Medicare / Medicaid / private / self-pay | Determines reimbursement and financial risk |
| Geography | County / service-area averages | Reveals regional capacity gaps (e.g. post-acute care) |

## Design decisions that follow from the research
1. **Median and percentiles next to means** - stay and charge distributions are right-skewed.
2. **Log scales** on charge plots - otherwise a handful of very expensive stays flatten every box.
3. **Severity is a first-class dimension** (sidebar filter, box plot, pivot) - the standard risk-adjustment lens.
4. **Charges and costs are shown separately**, with a cost-to-charge ratio, because charges are list prices.
5. **Minimum-case thresholds** for rankings - averages over 3 patients are noise.
6. **Outlier handling is explicit and reversible** (3xIQR on the log scale, sidebar toggle).
7. **Rank-based statistics** (Spearman, Kruskal-Wallis) because data are not normally distributed.

## Sources
- NYS Department of Health, SPARCS: https://www.health.ny.gov/statistics/sparcs/
- AHRQ HCUP Clinical Classifications Software Refined (CCSR): https://hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp
- Streamlit blog, "Improving healthcare management with Streamlit": https://blog.streamlit.io/improving-healthcare-management-with-streamlit/
- Plotly Express: https://plotly.com/python/plotly-express/
- Streamlit docs: https://docs.streamlit.io/
