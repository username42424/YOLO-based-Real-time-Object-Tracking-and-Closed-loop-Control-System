# -*- coding: utf-8 -*-
import io

p = r"C:\Users\12951\Desktop\12323\tests\test_tracking_v4.py"
s = io.open(p, encoding="utf-8").read()
n = 0

reps = [
    # timing_starts_at_first_detection
    ("""        rows = [_obs(1.0, vel=(600.0, 0.0))]
        out = self._agg(rows, _elig(streak=2.0, outside=True, fdt=0.8))
        self.assertEqual(out["eligible@0.3"], 1)""",
     """        rows = [_obs(1.0, vel=(600.0, 0.0))]
        rec = FakeRec(rows, [], {"first_inner60_time": 1.0,
                                 "post_entry_inner60_seconds": 0.0,
                                 "post_entry_observed_seconds": 1.0})
        out = aggregate_one_phase([rec], [_elig(streak=2.0, outside=True,
                                                fdt=0.8)], cap_counts=280.0)
        self.assertEqual(out["eligible@0.3"], 1)"""),
    # entry_at_absolute_1s test also needs fi (it asserts timeout) — same shape
    ("""        rows = [_obs(1.0, vel=(600.0, 0.0))]
        out = self._agg(rows, _elig(streak=2.0, outside=True, fdt=0.0))""",
     """        rows = [_obs(1.0, vel=(600.0, 0.0))]
        rec = FakeRec(rows, [], {"first_inner60_time": 1.0,
                                 "post_entry_inner60_seconds": 0.0,
                                 "post_entry_observed_seconds": 1.0})
        out = aggregate_one_phase([rec], [_elig(streak=2.0, outside=True,
                                                fdt=0.0)], cap_counts=280.0)"""),
    # timestamp inversion
    ("""        rows = [_obs(0.1, vel=(600.0, 0.0))]
        out = self._agg(rows, _elig(streak=2.0, outside=True, fdt=0.8))""",
     """        rows = [_obs(0.1, vel=(600.0, 0.0))]
        rec = FakeRec(rows, [], {"first_inner60_time": 0.1,
                                 "post_entry_inner60_seconds": 0.0,
                                 "post_entry_observed_seconds": 1.0})
        out = aggregate_one_phase([rec], [_elig(streak=2.0, outside=True,
                                                fdt=0.8)], cap_counts=280.0)"""),
    # median uses corrected elapsed
    ("""        rows = [_obs(0.8, vel=(600.0, 0.0))]
        out = self._agg(rows, _elig(streak=2.0, outside=True, fdt=0.6))""",
     """        rows = [_obs(0.8, vel=(600.0, 0.0))]
        rec = FakeRec(rows, [], {"first_inner60_time": 0.8,
                                 "post_entry_inner60_seconds": 0.0,
                                 "post_entry_observed_seconds": 1.0})
        out = aggregate_one_phase([rec], [_elig(streak=2.0, outside=True,
                                                fdt=0.6)], cap_counts=280.0)"""),
    # AB perfect-data assertion
    ("""        # steady state: velocity estimate close to 500, no gate trips
        self.assertGreater(abs(v - 500.0), 0.0)
        self.assertLess(abs(v - 500.0), 250.0)""",
     """        # perfect constant-velocity data: innovation is zero, the filter
        # must pass the true velocity through unchanged
        self.assertAlmostEqual(v, 500.0)"""),
]
for old, new in reps:
    if old in s:
        s = s.replace(old, new)
        n += 1
    else:
        print("NOT FOUND:", old.splitlines()[0])
io.open(p, "w", encoding="utf-8").write(s)
print("applied", n, "of", len(reps))
