# Tests

Run the checks in the project environment:

```bash
python -m unittest discover -s tests -v
```

The model checks cover MOMA feature grids and gradients, MTIH box and mask
outputs, three-class configuration, SAM teacher targets, cascade losses, and
inference. Earlier module names are also checked as import and registry aliases.

The evaluation checks cover exact detections, category-aware and category-agnostic
matching, and empty predictions using controlled COCO examples.
