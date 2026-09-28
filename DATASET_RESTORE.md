# Dataset restore guide

The datasets are stored as compressed archives to keep them within GitHub's
100M per-file limit.

## 200_train_cases

```bash
tar xzf 200_train_cases.tar.gz
```

## data (split into <100M volumes)

Reassemble the split volumes, verify the checksum, then extract:

```bash
cat data.tar.gz.*.part > data.tar.gz
sha256sum -c data.tar.gz.sha256
tar xzf data.tar.gz
```

The `.part` files are ordered by their numeric suffix (`00`..`05`); the shell
glob expands them in the correct order.
