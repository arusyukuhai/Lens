## Plateau-triggered rollback scheduler.
## Promotion is evaluated on disjoint acceptance source chunks.
## Protected heldout is NEVER consulted by this controller.
type
  RollbackControl* = object
    noPromotionStreak*: int
    lastAttempt*: int # 1-based iteration; zero means never attempted

proc recordPromotion*(ctl: var RollbackControl, promoted: bool) =
  if promoted:
    ctl.noPromotionStreak = 0
  else:
    inc ctl.noPromotionStreak

proc rollbackReason*(ctl: RollbackControl, iteration, rules,
                     interval, patience: int): string =
  ## Return a reason instead of a bool so skipped rollbacks can be audited.
  if interval <= 0: return "disabled"
  if rules < 2: return "too-few-rules"
  if ctl.noPromotionStreak < patience: return "progress-or-insufficient-stagnation"
  if ctl.lastAttempt > 0 and iteration-ctl.lastAttempt < interval:
    return "cooldown"
  "plateau"

proc recordAttempt*(ctl: var RollbackControl, iteration: int) =
  ctl.lastAttempt = iteration
