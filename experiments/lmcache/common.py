import os


def setup_lmcache_environment(
    *,
    chunk_size: int = 256,
    blend_special_str: str = "# #",
    blend_min_tokens: int = 256,
    local_cpu_size_gb: float = 50.0,
    recompute_ratios: float = 0.15,
    blend_check_layers: str = "1",
    enable_sparse: bool = False,
    enable_async_loading: bool = False,
) -> None:
    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_BLEND_MIN_TOKENS"] = str(blend_min_tokens)
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = blend_check_layers
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = str(recompute_ratios)
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(local_cpu_size_gb)
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = (
        "True" if enable_async_loading else "False"
    )
    # LMCache async loading requires fixed-size chunks only.
    os.environ["LMCACHE_SAVE_UNFULL_CHUNK"] = (
        "False" if enable_async_loading else "True"
    )

    os.environ["LMCACHE_LOG_LEVEL"] = "WARNING"
    # os.environ["LMCACHE_LOG_LEVEL"] = "INFO"



    # Explicitly clear any old disk-persistence settings so every run uses
    # LMCache's default CPU-only eviction behavior.
    for env_name in (
        "LMCACHE_LOCAL_DISK",
        "LMCACHE_MAX_LOCAL_DISK_SIZE",
        "PATHWEAVER_ENABLE_LMCACHE_PATCHES",
        "PATHWEAVER_LMCACHE_PATCH_DISK_CACHE_DIR",
        "PATHWEAVER_LMCACHE_WRITE_BUFFER",
        "PATHWEAVER_LMCACHE_WRITE_BUFFER_MAX_ITEMS",
        "PATHWEAVER_LMCACHE_WRITE_BUFFER_MAX_MB",
        "PATHWEAVER_LMCACHE_WRITE_BUFFER_FLUSH_MS",
        "PATHWEAVER_LMCACHE_WRITE_WORKERS",
        "PATHWEAVER_PRELOAD_DISK_KV_TO_CPU",
        "PATHWEAVER_PRELOAD_MAX_GB",
    ):
        os.environ.pop(env_name, None)

    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        os.environ["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'
    else:
        os.environ.pop("LMCACHE_EXTRA_CONFIG", None)


def get_blend_separator(default: str) -> str:
    return os.getenv("LMCACHE_BLEND_SPECIAL_STR", default)


def destroy_lmcache_engine() -> None:
    try:
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.v1.cache_engine import LMCacheEngineBuilder

        LMCacheEngineBuilder.destroy(ENGINE_NAME)
    except Exception as exc:
        print(f"⚠️ LMCache cleanup skipped: {exc}")
