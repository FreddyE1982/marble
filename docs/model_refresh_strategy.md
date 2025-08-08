# Model Refresh Strategy

When datasets evolve, Neuronenblitz models must determine whether to refresh
weights through a full retrain or an incremental update. The decision hinges on
several criteria.

## Criteria for Selecting Refresh Type
- **Scale of dataset change** – if more than ~20% of samples are new or
  distribution statistics shift beyond preset thresholds, perform a full
  retrain. Minor additions favour incremental updates.
- **Architecture compatibility** – incremental updates assume identical model
  structure. Any topology modification requires a full retrain.
- **Performance regression** – track validation loss after an incremental update;
  if loss fails to recover within a few epochs, escalate to full retrain.

## Strategy Comparison
### Full Retrain
Retrains the model from scratch using the latest dataset snapshot.
- **Pros:** guarantees consistency with new data, handles architecture changes.
- **Cons:** computationally expensive; discards previous training time.

### Incremental Update
Continues training from existing weights using only the new or modified data.
- **Pros:** faster, preserves learned features.
- **Cons:** risk of accumulating bias if data drift is large; incompatible with
  structural changes.

## API

The :mod:`model_refresh` module provides helpers for both strategies.  They
automatically use a GPU when available but fall back to CPU execution so the
same call works across hardware tiers.

```python
from model_refresh import full_retrain, incremental_update

# Retrain from scratch on the entire dataset
full_retrain(model, full_dataset, epochs=5)

# Continue training with only new samples
incremental_update(model, new_samples, epochs=2)
```
