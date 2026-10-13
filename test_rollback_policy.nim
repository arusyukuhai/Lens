import std/unittest
import lens_rollback

suite "Stagnation-driven rollback controller":
  test "improving acceptance never triggers periodic rollback":
    var c: RollbackControl
    for iteration in 1..25:
      c.recordPromotion(true)
      check rollbackReason(c,iteration,6,4,3) != "plateau"
    check c.lastAttempt == 0
    check c.noPromotionStreak == 0

  test "three rejections trigger rollback, and cooldown is enforced":
    var c: RollbackControl
    for iteration in 1..3:
      c.recordPromotion(false)
      if iteration < 3:
        check rollbackReason(c,iteration,6,4,3) != "plateau"
    check rollbackReason(c,3,6,4,3) == "plateau"
    c.recordAttempt(3)
    for iteration in 4..6:
      c.recordPromotion(false)
      check rollbackReason(c,iteration,6,4,3) == "cooldown"
    c.recordPromotion(false)
    check rollbackReason(c,7,6,4,3) == "plateau"

  test "successful promotion resets stagnation even after rollback":
    var c: RollbackControl
    for i in 1..5: c.recordPromotion(false)
    c.recordAttempt(5)
    c.recordPromotion(true)
    check c.noPromotionStreak == 0
    check rollbackReason(c,6,6,4,3) == "progress-or-insufficient-stagnation"

  test "disable and insufficient rules block expensive regrow":
    var c: RollbackControl
    for i in 1..10: c.recordPromotion(false)
    check rollbackReason(c,10,8,0,3) == "disabled"
    check rollbackReason(c,10,1,4,3) == "too-few-rules"
