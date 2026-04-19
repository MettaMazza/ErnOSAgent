use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;
    println!("Metal device OK");

    let nf = 131072usize;
    let md = 5376usize;
    let total = nf * md;
    println!("Generating {} random f32 values on Metal ({:.1} GB)...", total, total as f64 * 4.0 / 1e9);
    
    let t = Tensor::randn(0.0f32, 1.0, (nf, md), &device)?;
    
    let max_val = t.max_all()?.to_scalar::<f32>()?;
    let min_val = t.min_all()?.to_scalar::<f32>()?;
    let sum_val = t.abs()?.sum_all()?.to_scalar::<f32>()?;
    
    println!("randn: max={max_val}, min={min_val}, abs_sum={sum_val}");
    println!("  has_nan={}, has_inf={}", max_val.is_nan() || min_val.is_nan(), max_val.is_infinite() || min_val.is_infinite() || sum_val.is_infinite());
    
    let scale = (6.0 / (nf + md) as f64).sqrt();
    let scaled = (&t * scale)?;
    let s_max = scaled.max_all()?.to_scalar::<f32>()?;
    let s_min = scaled.min_all()?.to_scalar::<f32>()?;
    println!("Scaled (Xavier {:.6}): max={s_max}, min={s_min}", scale);
    
    // Simulate the actual matmul
    let x = Tensor::randn(0.0f32, 0.015, (95, md), &device)?;
    let result = x.matmul(&scaled.t()?)?;
    let r_max = result.max_all()?.to_scalar::<f32>()?;
    let r_min = result.min_all()?.to_scalar::<f32>()?;
    let r_sum = result.abs()?.sum_all()?.to_scalar::<f32>()?;
    println!("Matmul [95,{md}] x [{md},{nf}]: max={r_max}, min={r_min}, abs_sum={r_sum}");
    println!("  has_nan={}, has_inf={}", r_max.is_nan() || r_min.is_nan(), r_max.is_infinite() || r_min.is_infinite() || r_sum.is_infinite());
    
    Ok(())
}
