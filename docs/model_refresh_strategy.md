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
