# Faster-Whisper unload/reload benchmark: disk cache vs `move_to_cpu`

Date: 2026-08-20

Runtime:

- Container/service: `wlk-gpu-sortformer`
- Backend: `faster-whisper`
- Model: `large-v3`
- Language: `en`
- Strategy under test: `--model-unload-strategy=move_to_cpu`
- Cached model bytes for RAM path: `3,090,835,702` bytes (`2.88 GiB`)

Question:

Does the `move_to_cpu` implementation, which reloads faster-whisper through
`WhisperModel(..., files=...)`, actually reload faster than dropping the model
and letting faster-whisper/CTranslate2 reload from the local model cache?

Benchmark method:

Ran a small Python benchmark inside the running Docker container. Before the
benchmark, the live server model was unloaded via `POST /dev/unload` to reduce
GPU-memory interference.

The benchmark instantiated `FasterWhisperASR(lan="en", model_size="large-v3")`
directly and measured two modes:

1. `disk_drop`
   - `unload_model(keep_cpu_cache=False)`
   - `ensure_model_loaded()`
   - Reload source: normal local Hugging Face / faster-whisper model cache.

2. `ram_files`
   - `unload_model(keep_cpu_cache=True)`
   - This reads every faster-whisper model file into a Python `dict[str, bytes]`.
   - `ensure_model_loaded()`
   - Reload source: `WhisperModel(..., files=dict(self._model_files_cpu_cache))`.

Results:

| Mode | Reload avg | Reload min | Reload max | Unload avg |
| --- | ---: | ---: | ---: | ---: |
| Initial load | 4.76s | - | - | - |
| `disk_drop` | 3.67s | 3.60s | 3.74s | 0.17s |
| `ram_files` | 4.32s | 4.27s | 4.38s | 1.30s |

Observed per-cycle reloads:

- `disk_drop`: `3.67s`, `3.74s`, `3.60s`
- `ram_files`: `4.27s`, `4.33s`, `4.38s`

Conclusion:

On this machine, `move_to_cpu` is slower than the normal disk/cache reload path.
The likely reason is that the OS page cache is already keeping the model files
hot in RAM for the normal path. The `files=` path avoids filesystem reads, but
adds Python `bytes`/`dict` overhead and still has to reconstruct the
CTranslate2 model and reinitialize/copy weights for GPU execution.

Practical takeaway:

For faster-whisper on this setup, `move_to_cpu` does not improve reload latency:

- Normal drop/reload: about `3.67s`
- RAM files reload: about `4.32s`
- RAM files also adds about `1.1s` extra unload cost to copy `2.88 GiB` into
  Python-managed memory.

If optimizing reload speed further, focus on avoiding full model reconstruction
or keeping a backend-supported resident model object, not on staging model files
in Python RAM.

## Follow-up: CTranslate2 already supports CPU weight caching

After the `files=` benchmark above, we inspected the underlying CTranslate2
Whisper object exposed at `faster_whisper.WhisperModel.model`.

CTranslate2 provides:

```python
ct2_whisper.unload_model(to_cpu=True)
ct2_whisper.load_model(keep_cache=True)
```

Docstring summary:

- `unload_model(to_cpu=True)` unloads the model from the initial device while
  keeping enough runtime context to quickly resume. With `to_cpu=True`, the
  model is moved to CPU memory instead of fully unloaded.
- `load_model(keep_cache=True)` loads the model back to the initial device and
  keeps the CPU cache.

Benchmark inside the same running container using the patched
`FasterWhisperASR` implementation:

| Mode | Reload avg | Unload avg |
| --- | ---: | ---: |
| CTranslate2 `to_cpu=True` / `keep_cache=True` | 0.595s | 0.338s |

Per-cycle reloads:

- `0.595s`, `0.598s`, `0.592s`

Per-cycle unloads:

- `0.794s`, `0.109s`, `0.109s`

Conclusion:

The right `move_to_cpu` implementation for faster-whisper is not the Python
`files=` model-file cache. It is CTranslate2's built-in model cache:

```python
self.model.model.unload_model(to_cpu=True)
self.model.model.load_model(keep_cache=True)
```

This avoids reconstructing the Python `WhisperModel` and the CTranslate2 model
object, and only moves the CTranslate2 weights between the initial device and
CPU memory. On this machine that changes reload latency from roughly `3.67s`
for normal drop/reload and `4.32s` for `files=` reload to roughly `0.60s`.
