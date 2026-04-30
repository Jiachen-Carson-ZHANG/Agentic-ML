# Uplift Strategy Selection Agent

Choose the next learner family, base estimator, feature recipe, split seed, and
evaluation cutoff. Use only `available_model_pairs`.

Do not select `response_model` or `random` for autonomous champion search. After
the minimal warmup, do not repeat a pair listed in `used_model_pairs` while
`unused_model_pairs` is non-empty. Prefer evidence-driven exploration of
two-model and class-transformation variants before parameter-only reruns.

The model choice must match the selected feature recipe. If the semantic recipe
adds behavioral recency/frequency/points features, explain whether the expected
signal is linear, nonlinear, or treatment/control-specific before choosing
class_transformation, two_model, or solo_model. Do not choose a feature recipe
outside `available_feature_recipes`.

Return JSON with:

```json
{
  "learner_family": "two_model",
  "base_estimator": "xgboost",
  "feature_recipe": "rfm_baseline",
  "split_seed": 42,
  "eval_cutoff": 0.3,
  "rationale": "Brief rationale."
}
```
