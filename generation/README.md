# Generated Camera Results

Each inference run owns one directory here. Generation saves camera data first;
visualization and evaluation add derived artifacts later without rerunning the
models.

```text
generation/<run-name>/
|-- manifest.json
|-- config.yaml
|-- camera/
|-- keyframes/
|-- metrics/
|-- vis/
`-- vmd/
```

Inference creates `camera/`, `keyframes/`, the manifest and the resolved config.
The `vis/`, `metrics/`, and `vmd/` directories are created on demand by their
corresponding tools.
