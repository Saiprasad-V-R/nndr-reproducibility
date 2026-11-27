# T10 — Independent time-varying planted plane

Run without arguments from any working directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py
```

`Q_star(t)` is the independent planted truth. The center and end references, and
compressed arithmetic-operator window diagnostic, are all distinct. `M=1` is
nested in `M=12` using the same first recording. Invalid fitted thresholds
are missing, never clipped. Outputs include vector-PDF and PNG figure,
per-window/per-subject numerical data, and scientific report.
