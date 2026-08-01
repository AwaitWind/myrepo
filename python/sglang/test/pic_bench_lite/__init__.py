"""pic_bench_lite — client-side benchmark helpers ported from pic_bench.

Provides scorers (F1 / ROUGE-L / substring), dataset loaders, prompt builder
and warmup primitives so quick_test_online.py can measure real task accuracy
(not only the FDT diagnostic metric).
"""
