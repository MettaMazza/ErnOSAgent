// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
// Module: Per-layer LoRA weight initialisation

// ─── Original work by @mettamazza — do not remove this attribution ───
//! Base model weight loading and LoRA VarMap initialization.

use super::LoraConfig;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;

/// Load the base model weights from safetensors shards via mmap.
pub fn load_base_weights(weights_dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let index_path = weights_dir.join("model.safetensors.index.json");

    if index_path.exists() {
        load_multi_shard(weights_dir, &index_path, device)
    } else {
        load_single_shard(weights_dir, device)
    }
}

fn load_multi_shard(
    weights_dir: &Path,
    index_path: &Path,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let index_content = std::fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read weight index: {}", index_path.display()))?;
    let index: serde_json::Value = serde_json::from_str(&index_content)
        .context("Failed to parse model.safetensors.index.json")?;

    let shard_names: std::collections::BTreeSet<String> = index
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("weight_map not found in index")?
        .values()
        .filter_map(|v| v.as_str())
        .map(|s| s.to_string())
        .collect();

    let shard_paths: Vec<std::path::PathBuf> = shard_names
        .iter()
        .map(|name| weights_dir.join(name))
        .collect();

    tracing::info!(
        shards = shard_paths.len(),
        dir = %weights_dir.display(),
        "Loading base model weights (multi-shard mmap)"
    );

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&shard_paths, DType::BF16, device)
            .with_context(|| "Failed to mmap safetensors shards")?
    };
    Ok(vb)
}

fn load_single_shard(weights_dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let single = weights_dir.join("model.safetensors");
    if !single.exists() {
        bail!(
            "No model weights found in {}. Run scripts/download_weights.sh first.",
            weights_dir.display()
        );
    }
    tracing::info!(path = %single.display(), "Loading base model weights (single shard)");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[single], DType::BF16, device)
            .context("Failed to mmap single shard")?
    };
    Ok(vb)
}

/// Per-layer projection dimensions detected from the actual model weights.
#[derive(Debug, Clone)]
pub struct LayerDims {
    pub q_out: usize,
    pub k_out: usize,
    pub v_out: usize,  // 0 means K=V sharing (no separate v_proj)
    pub o_in: usize,
}

/// Detect per-layer projection dimensions from safetensors header.
///
/// Reads the actual weight shapes from the safetensors file headers so each
/// layer's LoRA adapters are initialised with the correct dimensions.
/// This handles architectures with heterogeneous layers (e.g. Gemma 4's
/// alternating sliding_attention / full_attention with different head dims).
pub fn detect_per_layer_dims(
    weights_dir: &Path,
    model_prefix: &str,
    num_layers: usize,
    hidden_dim: usize,
) -> Result<Vec<LayerDims>> {
    let index_path = weights_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        // No index — assume uniform layers from config
        let uniform = LayerDims {
            q_out: hidden_dim,
            k_out: hidden_dim,
            v_out: hidden_dim,
            o_in: hidden_dim,
        };
        return Ok(vec![uniform; num_layers]);
    }

    // Read safetensors headers to get actual shapes
    let shapes = read_safetensors_shapes(weights_dir)?;

    let mut layer_dims = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let lp = if model_prefix.is_empty() {
            format!("layers.{layer}")
        } else {
            format!("{model_prefix}.layers.{layer}")
        };

        let q_key = format!("{lp}.self_attn.q_proj.weight");
        let k_key = format!("{lp}.self_attn.k_proj.weight");
        let v_key = format!("{lp}.self_attn.v_proj.weight");
        let o_key = format!("{lp}.self_attn.o_proj.weight");

        // Weight shape is [out_dim, in_dim] for linear layers
        let q_out = shapes.get(&q_key)
            .map(|s| s[0])
            .unwrap_or(hidden_dim);
        let k_out = shapes.get(&k_key)
            .map(|s| s[0])
            .unwrap_or(hidden_dim);
        let v_out = shapes.get(&v_key)
            .map(|s| s[0])
            .unwrap_or(0); // 0 = K=V sharing
        let o_in = shapes.get(&o_key)
            .map(|s| s[1]) // o_proj: [hidden, q_dim] → in_dim = s[1]
            .unwrap_or(q_out);

        tracing::debug!(
            layer, q_out, k_out, v_out, o_in,
            "Detected layer projection dims"
        );

        layer_dims.push(LayerDims { q_out, k_out, v_out, o_in });
    }

    // Log the layer type distribution
    let uniform_count = layer_dims.iter()
        .filter(|d| d.q_out == layer_dims[0].q_out)
        .count();
    if uniform_count < num_layers {
        tracing::info!(
            uniform = uniform_count,
            variant = num_layers - uniform_count,
            "Detected heterogeneous layer types"
        );
    }

    Ok(layer_dims)
}

/// Read all tensor shapes from safetensors headers (no data loaded).
fn read_safetensors_shapes(weights_dir: &Path) -> Result<HashMap<String, Vec<usize>>> {
    let mut shapes = HashMap::new();

    for entry in std::fs::read_dir(weights_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "safetensors") {
            let file = std::fs::File::open(&path)?;
            let mut reader = std::io::BufReader::new(file);

            // Read 8-byte header length
            let mut len_buf = [0u8; 8];
            std::io::Read::read_exact(&mut reader, &mut len_buf)?;
            let header_len = u64::from_le_bytes(len_buf) as usize;

            // Read header JSON
            let mut header_buf = vec![0u8; header_len];
            std::io::Read::read_exact(&mut reader, &mut header_buf)?;
            let header: serde_json::Value = serde_json::from_slice(&header_buf)?;

            if let Some(obj) = header.as_object() {
                for (name, meta) in obj {
                    if name == "__metadata__" { continue; }
                    if let Some(shape_arr) = meta.get("shape").and_then(|s| s.as_array()) {
                        let shape: Vec<usize> = shape_arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                            .collect();
                        shapes.insert(name.clone(), shape);
                    }
                }
            }
        }
    }

    tracing::debug!(tensor_count = shapes.len(), "Read safetensors shapes");
    Ok(shapes)
}

/// Build the LoRA VarMap with per-layer correct dimensions.
///
/// Each layer's LoRA A and B matrices are sized to match the actual model
/// weight shapes, ensuring full coverage across all layer types including
/// architectures with heterogeneous layers.
pub fn build_lora_varmap(config: &LoraConfig, device: &Device) -> Result<VarMap> {
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, device);
    let in_dim = config.hidden_dim();

    // Detect per-layer dimensions from the actual model weights
    let per_layer = detect_per_layer_dims(
        &config.weights_dir,
        &config.model_prefix,
        config.num_layers(),
        in_dim,
    )?;

    let mut total_params = 0usize;

    for (layer, dims) in per_layer.iter().enumerate() {
        for module in &config.target_modules {
            let prefix = format!("lora.{layer}.{module}");

            // Determine the correct output dimension for this module on this layer
            let out_dim = match module.as_str() {
                "q_proj" => dims.q_out,
                "k_proj" => dims.k_out,
                "v_proj" => {
                    if dims.v_out == 0 {
                        // K=V sharing — LoRA v_proj uses same dims as k_proj
                        dims.k_out
                    } else {
                        dims.v_out
                    }
                }
                "o_proj" => in_dim, // o_proj always outputs hidden_dim
                _ => in_dim,
            };

            let bound = (6.0_f64 / in_dim as f64).sqrt();
            // A: [rank, in_dim] — down-projection from input space
            vs.pp(&prefix).get_with_hints(
                (config.rank, in_dim),
                "lora_a",
                candle_nn::init::Init::Uniform { lo: -bound, up: bound },
            )?;
            // B: [out_dim, rank] — up-projection to output space
            vs.pp(&prefix).get_with_hints(
                (out_dim, config.rank),
                "lora_b",
                candle_nn::init::Init::Const(0.0),
            )?;

            total_params += config.rank * in_dim + config.rank * out_dim;
        }
    }

    tracing::info!(
        layers = config.num_layers(),
        modules = config.target_modules.len(),
        rank = config.rank,
        total_params,
        "LoRA VarMap built on {:?}",
        device
    );

    Ok(var_map)
}

/// Load a previous adapter's weights from a safetensors file.
///
/// Returns a map of tensor name → Tensor so we can initialise
/// the new VarMap from the previous training run's weights.
pub fn load_previous_adapter(
    adapter_dir: &Path,
    device: &Device,
) -> Result<HashMap<String, candle_core::Tensor>> {
    let adapter_path = adapter_dir.join("adapter_model.safetensors");
    if !adapter_path.exists() {
        anyhow::bail!(
            "Previous adapter not found at {}",
            adapter_path.display()
        );
    }

    let tensors = candle_core::safetensors::load(&adapter_path, device)
        .with_context(|| format!(
            "Failed to load previous adapter: {}",
            adapter_path.display()
        ))?;

    tracing::info!(
        tensors = tensors.len(),
        path = %adapter_path.display(),
        "Previous adapter loaded for stacking"
    );

    Ok(tensors)
}

/// Build a LoRA VarMap initialised from a previous adapter's weights.
///
/// This is the core mechanism for cumulative adapter stacking — instead of
/// initialising B matrices to zero (losing all previous training), we load
/// the previous adapter's trained weights as the starting point.
pub fn build_lora_varmap_with_resume(
    config: &LoraConfig,
    device: &Device,
    resume_from: &Path,
) -> Result<VarMap> {
    // First, load the previous adapter's weights
    let previous = load_previous_adapter(resume_from, device)?;

    // Build the VarMap normally (with random A / zero B init)
    let var_map = build_lora_varmap(config, device)?;

    // Overwrite matching tensors from the previous adapter
    let mut resumed = 0usize;
    let data = var_map.data().lock().map_err(|e| {
        anyhow::anyhow!("VarMap lock failed: {e}")
    })?;

    for (name, var) in data.iter() {
        if let Some(prev_tensor) = previous.get(name) {
            // Shape must match — skip silently if architecture changed
            if var.as_tensor().dims() == prev_tensor.dims() {
                var.set(prev_tensor)?;
                resumed += 1;
            } else {
                tracing::debug!(
                    tensor = %name,
                    prev_shape = ?prev_tensor.dims(),
                    new_shape = ?var.as_tensor().dims(),
                    "Shape mismatch — skipping resume for this tensor"
                );
            }
        }
    }

    tracing::info!(
        resumed = resumed,
        total = data.len(),
        adapter_path = %resume_from.display(),
        "LoRA VarMap resumed from previous adapter"
    );

    drop(data);
    Ok(var_map)
}
