fn main() {
    let content = "This is a test with an emoji 🌿 and some text.";
    let max_len = 15;
    let chunks = chunk_message(content, max_len);
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: '{}'", i, chunk);
    }
}

pub(crate) fn chunk_message(content: &str, max_len: usize) -> Vec<String> {
    if content.len() <= max_len {
        return vec![content.to_string()];
    }
    let mut chunks = Vec::new();
    let mut remaining = content;
    while !remaining.is_empty() {
        let split_at = if remaining.len() <= max_len {
            remaining.len()
        } else {
            // Find a valid UTF-8 char boundary at or before max_len
            let boundary = {
                let mut b = max_len;
                while b > 0 && !remaining.is_char_boundary(b) {
                    b -= 1;
                }
                b
            };
            remaining[..boundary]
                .rfind('\n')
                .map(|idx| idx + 1) // include the newline in the current chunk
                .unwrap_or(boundary)
        };
        let (chunk, rest) = remaining.split_at(split_at);
        chunks.push(chunk.to_string());
        remaining = rest.trim_start_matches('\n');
    }
    chunks
}
