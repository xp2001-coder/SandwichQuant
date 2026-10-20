# Security and local artifacts

- Use trusted local model directories. The legacy loaders enable
  `trust_remote_code=True`; local custom model files can execute Python.
- Some existing backend and trainer checkpoints are loaded using
  `torch.load(..., weights_only=False)`. Untrusted pickle files can execute
  arbitrary code. Do not use these paths for downloaded untrusted checkpoints.
- Review any external backend before placing it on the import path.
- Keep credentials in the process environment or an ignored local file.
  Do not put credentials in configuration examples or command-line arguments
  that will be logged.
- Generated manifests and logs can contain local paths, hardware details and
  command arguments. Review them before sharing; they are not source-release
  assets.
- The supplied framework PNG is a documentation asset, not executable content.
