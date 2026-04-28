# Paper Outline — XPrice: An Explainable AI Framework for Transparent Dynamic Pricing
## MIT 622 Final Project | Group 1
## Target: 1,800–2,200 words | APA 7th | Min. 10 academic references

---

## TITLE
**XPrice: An Explainable AI Framework for Transparent Dynamic Pricing in Urban Ride-Hailing — A Dubai Case Study**

---

## ABSTRACT (~170 words)
*Purpose:* Dynamic pricing algorithms in ride-hailing platforms operate as black boxes, generating user distrust and limiting operational intelligence for managers. This paper proposes XPrice, an Explainable AI (XAI) framework that applies SHAP (SHapley Additive exPlanations) to make surge pricing decisions interpretable for both internal operations teams and end-users at the point of booking.

*Methodology:* Using Design Science Research (Peffers et al., 2007), we construct a high-fidelity synthetic mirror dataset of 165,000 Dubai ride-hailing records across 20 pricing zones, resolve endpoints through 134 mapped Dubai community polygons, train an XGBoost price prediction model, and apply SHAP-style tree contributions to generate local and global explanations.

*Findings:* The framework reveals that distance, product class, demand pressure, and toll exposure are the dominant pricing drivers, with traffic and airport context as secondary effects. A live interactive application demonstrates real-time explainable price estimation with draggable pins, visible neighborhood boundaries, and plain-language breakdowns.

*Contribution:* XPrice advances both managerial decision-making and end-user transparency in algorithmic pricing — an underexplored intersection in the MENA ride-hailing context.

**Keywords:** Explainable AI, SHAP, Dynamic Pricing, Ride-Hailing, Decision Support Systems, Algorithmic Transparency, Business Analytics

---

## SECTION 1 — Introduction and Research Context (~250 words)

**Opening argument:**
Careem, operating across MENAP with 50M+ users, uses ML-driven dynamic pricing across its ride-hailing, food delivery, and mobility products. Yet like all major platforms, the algorithm that determines whether a user pays AED 18 or AED 46 for an identical route is entirely opaque — to the user who books the ride, and often to the operations manager who manages the market.

**The problem — two dimensions:**
1. *Internal:* Operations teams receive completion rate data and GMV metrics but cannot identify *which pricing factors* are creating supply gaps or undermining captain acceptance. This was evidenced in our Group Project: the 84% completion rate gap was observable but its pricing-side causes were not decomposable.
2. *External:* Riders experience price surges during events (Dubai Shopping Festival, GITEX), bad weather, or Ramadan without any explanation. Research shows algorithmic pricing erodes consumer trust when it is unexplained [R8, R9].

**Research question:**
*How can Explainable AI methods, specifically SHAP, be applied to ride-hailing dynamic pricing to improve transparency for both operations managers and end-users?*

**Significance:**
The EU AI Act (2024) and UAE's National AI Strategy both flag algorithmic transparency as a governance priority. In ride-hailing, regulators in the UK, EU, and GCC have begun scrutinizing surge pricing [R10, R15]. XAI provides the technical mechanism to meet these demands without abandoning ML-based pricing.

**Paper structure:** Brief roadmap of sections.

*Cite: [R8], [R9], [R10], [R15]*

---

## SECTION 2 — Literature Review and Theoretical Background (~400 words)

### 2.1 Dynamic Pricing in Ride-Hailing
Ride-hailing platforms use demand-supply algorithms to adjust fares in real time. Sun et al. (2020) model dynamic pricing as a function of supply capacity and demand fluctuation [R14]. Castillo et al. (2024) in *Management Science* show that pricing algorithms that optimize short-term revenue can create systemic supply shortfalls — the "wild goose chase" problem [R13]. ML approaches including gradient boosting and deep learning have become standard for price modeling [R11, R12].

The challenge: these models trade interpretability for accuracy. A boosted ensemble of thousands of decision trees predicts price well but cannot tell a manager *why* a price is what it is.

### 2.2 Explainable AI: SHAP and LIME
SHAP (Lundberg & Lee, 2017) applies Shapley values from cooperative game theory to attribute each feature's contribution to a model's output [R1]. Unlike earlier methods, SHAP provides both local explanations (why did *this* ride cost AED 38?) and global explanations (which features matter most across all rides?). Salih et al. (2025) confirm SHAP's superiority over LIME for tabular data: SHAP is consistent, globally coherent, and faster on tree models via TreeExplainer [R3].

LIME (Ribeiro et al., 2016) fits a local surrogate linear model around each prediction [R2]. It is useful for comparison but less stable and limited to local scope.

Arrieta et al. (2020) provide the theoretical taxonomy: XAI methods are either ante-hoc (interpretable by design) or post-hoc (explanation added after training). SHAP is post-hoc model-agnostic, making it compatible with any ML pipeline without retraining [R4].

### 2.3 XAI in Decision Support Systems
XAI-powered DSS are increasingly recognized as essential for high-stakes organizational decisions [R5, R6]. Research shows that managers are more confident in ML recommendations when they receive feature-level explanations [R5]. XAI integration into DSS does not reduce predictive accuracy while improving user trust and model debugging speed [R6, R7].

### 2.4 Algorithmic Pricing Transparency and Consumer Trust
Zeithammer (2024) documents that consumers respond negatively to unexplained price changes, particularly when they perceive the algorithm as acting against their interests [R10]. Empirical studies confirm that transparent algorithmic pricing — where the rationale is disclosed — significantly improves consumer trust and purchase intention [R8, R9].

### 2.5 Regulatory Context
The EU AI Act (2024) classifies algorithmic decision-making systems affecting consumer economic interests as high-risk, requiring explainability and auditability [R15, R16]. GDPR's Article 22 establishes a "right to explanation" for automated decisions. UAE's National AI Strategy 2031 and Saudi Vision 2030 both emphasize ethical, trustworthy AI as a policy priority — creating a regulatory tailwind for XAI adoption in MENA platforms.

*Cite: [R1], [R2], [R3], [R4], [R5], [R6], [R7], [R8], [R9], [R10], [R13], [R14], [R15], [R16]*

---

## SECTION 3 — Critical Evaluation of the Proposed Solution (~250 words)

**What the Group Project built:**
An 8-page Streamlit supply-demand intelligence dashboard using 499,973 synthetic ride records across 5 MENAP cities. It covered descriptive → diagnostic → predictive → prescriptive analytics layers and identified a 3pp completion gap (84% vs 87% target), with recommendations for captain incentives and loyalty redesign.

**Strengths:**
- Strong descriptive and diagnostic coverage (completion funnel, cancellation reasons, city heatmaps)
- Multi-city scope with Ramadan overlay
- Scenario simulation (Pricing Lab)
- Clear business impact quantification (AED 0.88M GMV gap)

**Weaknesses — the XAI gap:**
1. *The dashboard showed WHAT happened, not WHY prices drove those outcomes.* The Pricing Lab used manual sliders — an analyst changed a surge multiplier and saw GMV change, but received no decomposition of which *input factors* warranted that multiplier.
2. *No end-user transparency layer.* The dashboard was purely internal. A rider on the Careem app seeing AED 42 has no insight into why.
3. *No predictive pricing model.* The "predictive" layer showed historical demand curves, not a trained price forecasting model.
4. *Static surge logic.* Surge was rule-based in the synthetic data, not learned from features — meaning SHAP could not be applied to explain it.

**Conclusion:** The Group Project established operational awareness. XPrice closes the explanation gap by adding an ML pricing model with SHAP decomposition and a user-facing explanation interface — moving from *observation* to *understanding*.

---

## SECTION 4 — Enhancement and Improvement: The XPrice Framework (~300 words)

**Framework overview:**
XPrice has three layers: (1) a feature-rich ML pricing model, (2) a SHAP explanation engine, and (3) a dual-audience application.

**4.1 Mirror Dataset Construction**
To demonstrate the proposed framework, we construct a synthetic mirror dataset of 165,000 Dubai ride records with 72 raw columns, 20 pricing zones, and neighborhood labels resolved from 134 mapped Dubai community polygons. The feature set covers weather (temperature, rain, sandstorm), active events (DSF, GITEX, NYE, Eid), temporal factors (hour, Ramadan, UAE holidays, peak), product type, route geometry, toll exposure, and supply-demand signals. Pricing is computed using a parametric formula calibrated to Careem's documented base fares and surge behavior (Careem Engineering, 2023). This researcher-constructed dataset is explicitly disclosed as synthetic — used to validate the methodology, following DSR practice (Peffers et al., 2007) [R17].

**4.2 ML Price Prediction Model**
We train an XGBoost regressor to predict `final_price_aed`. XGBoost is selected because its native `pred_contribs=True` API computes exact tree contribution values (equivalent to Shapley values) in polynomial time [R1, R3]. The refreshed model achieves R²=0.9880, RMSE=AED 5.56, MAE=AED 3.23, and CV R²=0.9878±0.0003 on a held-out block-split test set (n=29,993 rides) — substantially above our R²>0.92 threshold. Cyclical encoding of hour and day-of-week, and interaction features (`distance_x_traffic`, `traffic_x_peak`, `efficiency_x_traffic`), improve both accuracy and SHAP interpretability.

**4.3 SHAP Explanation Layer**
- *Global explanations (Operations Manager):* SHAP beeswarm and bar plots rank features by their mean absolute tree contribution across all rides. The top global drivers are: (1) `distance_x_traffic` — AED 14.90 mean contribution, (2) `route_distance_km` — AED 14.67, (3) `is_hala_product` — AED 14.26, and (4) `demand_index` — AED 4.96. Product-type effects and `salik_gates` remain strong secondary drivers, confirming that trip geometry and product/toll structure dominate before weather effects meaningfully enter the quote.
- *Local explanations (End-User):* For every individual ride, a SHAP waterfall chart shows the additive contribution of each feature to that specific fare. This is then translated into a plain-language sentence: *"Your fare includes a AED 3.20 surcharge because you're departing near a major event (GITEX) during peak hours."*

**4.4 SHAP vs LIME Comparison**
LIME is applied as a methodological comparison. SHAP proves more stable across repeated runs and provides globally consistent feature rankings, consistent with Salih et al. (2025) [R3].

*Cite: [R1], [R3], [R5], [R6], [R17]*

---

## SECTION 5 — Implementation and Business Impact (~250 words)

**Technical feasibility:**
XPrice is built on open-source tools (XGBoost, Streamlit, Folium, OpenWeatherMap API) — zero licensing cost. Shapley values are computed via XGBoost's native `pred_contribs=True` booster API, eliminating the external `shap` library dependency and reducing per-prediction latency. The application runs in real-time: a price + SHAP explanation generates in under 200ms, well within Careem's documented API latency requirements (Careem Engineering, 2020) [R18]. Integration into the existing YODA ML platform would require adding SHAP as a post-processing step to the deployed price prediction service.

**Organizational considerations:**
Two audiences require different explanation interfaces:
- *Operations managers* need global SHAP dashboards filterable by city, time, and event — integrated into their existing analytics tooling.
- *End-users* need a simplified, plain-language explanation at the moment of booking — requiring a front-end API endpoint that returns a SHAP-generated text alongside the price estimate.

The framework is also compatible with Careem's Galileo A/B testing infrastructure: explanation quality (user comprehension, trust scores) can be A/B tested across rider cohorts.

**Business impact:**
1. *Trust and retention:* Transparent pricing reduces fare-related cancellations. Research shows that price explanation improves purchase intention and reduces price sensitivity [R8, R9].
2. *Regulatory readiness:* XPrice provides an audit trail for every pricing decision — directly addressing EU AI Act obligations and UAE AI governance requirements [R15, R16].
3. *Operational intelligence:* Operations managers can identify which factors (events, weather, supply) are dominating pricing decisions in specific zones — enabling proactive captain dispatching and incentive targeting.
4. *Reduced complaints:* A major cost driver in ride-hailing customer service is price dispute resolution. Upfront SHAP-based explanations reduce disputes at source.

*Cite: [R8], [R9], [R15], [R16], [R18]*

---

## SECTION 6 — Reflection and Learning (~100 words)

**Key insights:**
- The transition from descriptive dashboards to explainable ML models requires a shift in both methodology (DSR vs exploratory analysis) and tooling (XGBoost + SHAP vs Pandas + Streamlit charts).
- Constructing a realistic synthetic dataset is itself a research contribution — it forces explicit documentation of pricing assumptions that are usually implicit in industry practice.
- Dual-audience design (internal ops vs external user) is architecturally and conceptually non-trivial: explanations that are informative to a data scientist are not useful to a rider.

**Future improvements:** Real-time model retraining, multi-city extension, integration of captain supply signals from live GPS data.

---

## SECTION 7 — Recommendations (~100 words)

1. **Adopt SHAP-based price explanations for the Careem app** — display a simplified SHAP breakdown at the fare estimation screen, starting with the top 3 contributing factors.
2. **Build an internal XAI Operations Dashboard** — replace the current static surge reports with a SHAP-powered feature decomposition tool for city operations managers.
3. **Establish an XAI governance policy** — document which features are permissible inputs to the pricing model (excluding protected attributes) and generate quarterly SHAP audits for regulatory compliance.
4. **Pilot in Dubai first** — Dubai's well-mapped zones, diverse event calendar, and tech-savvy user base make it the optimal pilot market.

---

## SECTION 8 — Conclusion (~100 words)

XPrice demonstrates that applying SHAP to a ride-hailing pricing model is technically feasible, computationally efficient, and organizationally valuable. The framework closes the explanation gap identified in the Group Project's supply-demand dashboard by moving from descriptive observation to algorithmic transparency. By making pricing understandable to both the operations manager (global SHAP analysis) and the rider (local waterfall explanation), XPrice advances the case for Explainable AI as a standard component of platform pricing architecture — particularly in MENA markets where regulatory transparency requirements are accelerating. The live application demonstrates that this is not theoretical: it works in real time, and it works for real users.

---

## WORD COUNT TARGET BY SECTION

| Section | Target Words |
|---------|-------------|
| Abstract | 170 |
| Keywords | — |
| 1. Introduction | 250 |
| 2. Literature Review | 400 |
| 3. Critical Evaluation | 250 |
| 4. Enhancement | 300 |
| 5. Implementation | 250 |
| 6. Reflection | 100 |
| 7. Recommendations | 100 |
| 8. Conclusion | 100 |
| **Total** | **~1,920** |

*(Add ~150–280 words distributed across section transitions and exhibit captions to reach 1,800–2,200 range)*

---

## CITATIONS MAP — Which reference goes where

| Reference | Sections Used |
|-----------|--------------|
| [R1] Lundberg & Lee 2017 (SHAP) | 2, 4 |
| [R2] Ribeiro et al. 2016 (LIME) | 2, 4 |
| [R3] Salih et al. 2025 (SHAP vs LIME) | 2, 4 |
| [R4] Arrieta et al. 2020 (XAI survey) | 2 |
| [R5] XAI-DSS Framework 2025 | 2, 4 |
| [R6] XAI-DSS Review MDPI 2024 | 2, 4 |
| [R7] XAI trustworthy framework 2025 | 2 |
| [R8] Ethics & Trust in AI Pricing 2026 | 1, 2, 5 |
| [R9] Algorithmic pricing trust 2024 | 1, 2, 5 |
| [R10] Zeithammer UCLA 2024 | 1, 2 |
| [R13] Castillo et al. Management Science 2024 | 2 |
| [R14] Sun et al. dynamic pricing 2020 | 2 |
| [R15] XAI GDPR framework MDPI 2025 | 1, 2, 5 |
| [R16] XAI AI Act framework MDPI 2025 | 2, 5 |
| [R17] Peffers et al. DSR 2007 | 3, 4 |
| [R18] Careem Engineering 2020 | 5 |
