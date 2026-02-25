# DDP Safety Audit Report

## Executive Summary
All DDP collective calls have been audited and fixed to ensure identical execution order across all ranks. The script is now DDP-safe and should not experience NCCL timeout errors.

## Collective Call Order (Per Epoch)

**ALL ranks execute these collectives in EXACTLY this order every epoch:**

1. `dist.all_reduce(avg_train_loss, op=SUM)` - Line ~1556
2. `dist.all_reduce(avg_val_loss, op=SUM)` - Line ~1588
3. `dist.all_reduce(total_correct, op=SUM)` - Line ~1593
4. `dist.all_reduce(total_samples_t, op=SUM)` - Line ~1594
5. `dist.all_reduce(cm_local, op=SUM)` - Line ~1597
6. `dist.broadcast(stop_tensor, src=0)` - Line ~1756

**Critical Fix:** The `dist.broadcast()` call was previously inside `if rank == 0:` block, meaning only rank 0 called it. This has been fixed - ALL ranks now call `dist.broadcast()`.

## DDP Safety Guarantees

### ✅ Identical Collective Order
- All collectives are executed unconditionally (not inside `if rank == 0:` blocks)
- No early returns that skip collectives on some ranks
- All collectives are synchronous (no async operations)

### ✅ Identical Batch Counts
- `drop_last=True` on train and val DataLoaders ensures identical batch counts
- `DistributedSampler` ensures proper data distribution
- Assertions added to verify non-empty loaders on all ranks

### ✅ Defensive Checks Added
- `assert len(train_loader) > 0` and `assert len(val_loader) > 0` on all ranks
- Per-rank batch count logging at initialization and epoch 1
- Debug environment variable hints documented

### ✅ Early Stopping Broadcast
- Rank 0 computes `should_stop` decision
- **ALL ranks** call `dist.broadcast(stop_tensor, src=0)` (fixed from previous bug)
- No extra barriers after broadcast (broadcast is already synchronous)

## Key Changes Made

### 1. Fixed Broadcast Bug (CRITICAL)
**Before:** Only rank 0 called `dist.broadcast()` (inside `if rank == 0:` block)
**After:** ALL ranks call `dist.broadcast()`, rank 0 sets the value

### 2. Replaced Heavy all_gather_object
**Before:** `dist.all_gather_object()` gathering millions of predictions/labels
**After:** Build 19x19 confusion matrix locally, then `dist.all_reduce(SUM)`
**Impact:** ~1000x reduction in data transferred (361 integers vs millions)

### 3. Added drop_last=True
**Before:** DataLoaders could have different batch counts across ranks
**After:** `drop_last=True` ensures identical batch counts
**Impact:** Prevents NCCL timeout from mismatched iteration lengths

### 4. Replaced Problematic Custom Sampler
**Before:** `DistributedClassBalancedSampler` put everything in class 0, causing uneven iterations
**After:** Standard `DistributedSampler` ensures identical batch counts
**Impact:** Eliminates root cause of uneven iteration lengths

### 5. Removed Unnecessary Barriers
**Before:** Multiple `dist.barrier()` calls after collectives
**After:** Removed barriers - `all_reduce` and `broadcast` are already synchronous
**Impact:** Reduces deadlock risk

### 6. Fixed Model Output
**Before:** Model output softmax probabilities
**After:** Model outputs logits, softmax applied only when needed
**Impact:** Better numerical stability, proper loss computation

### 7. Fixed Loss Function Selection
**Before:** `use_log_softmax` flag ignored, always used `FocalLoss` (expects probabilities)
**After:** Respects `use_log_softmax` flag, uses `FocalLossLogSoftmax` for logits
**Impact:** Correct loss computation with logits

### 8. Fixed DDP device_ids
**Before:** `device_ids=[rank % num_gpus]` (could mismatch with local_rank)
**After:** `device_ids=[local_rank]` (consistent with `torch.cuda.set_device()`)
**Impact:** Correct GPU assignment per rank

### 9. Fixed Confusion Matrix Visualization
**Before:** `range(20)` for 19 classes
**After:** `range(19)` for 19 classes
**Impact:** Correct axis labels

## Debug Environment Variables

For troubleshooting NCCL issues, set these before running:
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch distributed debug
export NCCL_DEBUG=INFO                  # NCCL debug info
export NCCL_DEBUG_SUBSYS=ALL            # All NCCL subsystems
```

## Verification Checklist

- [x] All collectives executed unconditionally (not in `if rank == 0:` blocks)
- [x] No early returns that skip collectives
- [x] Identical batch counts across ranks (drop_last=True + DistributedSampler)
- [x] All ranks call dist.broadcast() for early stopping
- [x] No unnecessary barriers after collectives
- [x] Defensive asserts for DataLoader lengths
- [x] Per-rank batch count logging
- [x] Confusion matrix all_reduce instead of all_gather_object
- [x] Model outputs logits (not softmax)
- [x] Loss function respects use_log_softmax flag
- [x] DDP device_ids uses local_rank

## Expected Behavior

1. **Epoch 1:** All ranks print their batch counts (should be identical)
2. **Every Epoch:** All ranks execute the same 6 collectives in the same order
3. **Early Stopping:** All ranks participate in broadcast, all break together
4. **No NCCL Timeout:** Identical collective order + lightweight collectives = no timeout

## Testing Recommendations

1. Run with 4 ranks and verify all ranks print identical batch counts
2. Monitor logs for "[DDP AUDIT]" messages to verify synchronization
3. Check that no rank times out during collectives
4. Verify early stopping works correctly (all ranks stop together)
