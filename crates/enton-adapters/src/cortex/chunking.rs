/// Minimum character length required before emitting an early first clause to TTS.
///
/// Kokoro is a neural TTS model whose phonemizer and prosody predictor require
/// sufficient phonetic context to generate natural pitch contours and prevent
/// choppy, clipped audio on short fragments (e.g. "Oi," or "Sim,"). A threshold
/// of 20 characters provides 3 to 5 words of prosodic context while enabling
/// sub-second time-to-first-audio (TTFA).
pub const MIN_FIRST_CLAUSE_CHARS: usize = 20;

/// Known abbreviations that should not trigger sentence termination when ending in a dot.
const KNOWN_ABBREVIATIONS: &[&str] = &[
    "dr", "dra", "sr", "sra", "prof", "profa", "ex", "etc", "vs", "p.ex", "mr", "mrs", "ms",
];

/// Checks if a dot or comma at `byte_idx` in `text` is part of a decimal number (e.g. "3,5" or "3.14").
fn is_decimal_separator(text: &str, byte_idx: usize) -> bool {
    let bytes = text.as_bytes();
    if byte_idx == 0 || byte_idx + 1 >= bytes.len() {
        return false;
    }
    let prev = bytes.get(byte_idx.saturating_sub(1)).copied();
    let next = bytes.get(byte_idx + 1).copied();
    match (prev, next) {
        (Some(p), Some(n)) => p.is_ascii_digit() && n.is_ascii_digit(),
        _ => false,
    }
}

/// Checks if the dot at `dot_byte_pos` in `text` belongs to a known abbreviation.
fn is_abbreviation_dot(text: &str, dot_byte_pos: usize) -> bool {
    let Some(prefix) = text.get(..dot_byte_pos) else {
        return false;
    };
    let mut word_start = dot_byte_pos;
    for (idx, ch) in prefix.char_indices().rev() {
        if ch.is_alphabetic() || ch == '.' {
            word_start = idx;
        } else {
            break;
        }
    }
    if word_start >= dot_byte_pos {
        return false;
    }
    let Some(word) = prefix.get(word_start..dot_byte_pos) else {
        return false;
    };
    let lower = word.to_lowercase();
    KNOWN_ABBREVIATIONS.contains(&lower.as_str())
}

#[derive(Debug)]
struct BoundaryMatch {
    final_end_pos: usize,
    remainder_offset: usize,
    #[expect(dead_code, reason = "retained for debugging and inspection")]
    is_clause: bool,
}

#[expect(
    clippy::too_many_lines,
    reason = "scans clauses, sentences, decimals, abbreviations, and quotes in a single pass"
)]
fn find_boundary(buffer: &str, search_start: usize, allow_clause: bool) -> Option<BoundaryMatch> {
    let slice = buffer.get(search_start..)?;
    let mut chars = slice.char_indices().peekable();

    while let Some((rel_idx, ch)) = chars.next() {
        let abs_pos = search_start + rel_idx;

        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            if ch == '.' && is_decimal_separator(buffer, abs_pos) {
                continue;
            }
            if ch == '.' && is_abbreviation_dot(buffer, abs_pos) {
                continue;
            }

            let mut end_pos = abs_pos + ch.len_utf8();
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch == '.' || next_ch == '!' || next_ch == '?' {
                    end_pos += next_ch.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }

            let mut final_end_pos = end_pos;
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch == '"'
                    || next_ch == '\''
                    || next_ch == '”'
                    || next_ch == '’'
                    || next_ch == ')'
                    || next_ch == ']'
                {
                    final_end_pos += next_ch.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }

            let is_terminal = if let Some(trailing) = buffer.get(final_end_pos..) {
                trailing.chars().next().is_none_or(char::is_whitespace)
            } else {
                true
            };

            if is_terminal {
                let remainder_offset = buffer
                    .get(final_end_pos..)
                    .and_then(|rem| {
                        rem.char_indices()
                            .find(|(_, c)| !c.is_whitespace())
                            .map(|(offset, _)| final_end_pos + offset)
                    })
                    .unwrap_or(buffer.len());

                return Some(BoundaryMatch {
                    final_end_pos,
                    remainder_offset,
                    is_clause: false,
                });
            }
        } else if allow_clause && (ch == ',' || ch == ';' || ch == ':' || ch == '—') {
            if ch == ',' && is_decimal_separator(buffer, abs_pos) {
                continue;
            }

            let end_pos = abs_pos + ch.len_utf8();
            let mut final_end_pos = end_pos;
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch == '"'
                    || next_ch == '\''
                    || next_ch == '”'
                    || next_ch == '’'
                    || next_ch == ')'
                    || next_ch == ']'
                {
                    final_end_pos += next_ch.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }

            let is_valid_clause = if ch == '—' {
                true
            } else if let Some(trailing) = buffer.get(final_end_pos..) {
                trailing.chars().next().is_none_or(char::is_whitespace)
            } else {
                true
            };

            if is_valid_clause {
                let candidate_chars = buffer
                    .get(..final_end_pos)
                    .map_or(0, |s| s.trim().chars().count());

                if candidate_chars >= MIN_FIRST_CLAUSE_CHARS {
                    let remainder_offset = buffer
                        .get(final_end_pos..)
                        .and_then(|rem| {
                            rem.char_indices()
                                .find(|(_, c)| !c.is_whitespace())
                                .map(|(offset, _)| final_end_pos + offset)
                        })
                        .unwrap_or(buffer.len());

                    return Some(BoundaryMatch {
                        final_end_pos,
                        remainder_offset,
                        is_clause: true,
                    });
                }
            }
        }
    }

    None
}

fn extract_completed_chunks_inner(
    buffer: &mut String,
    first_chunk_sent: &mut bool,
    allow_first_clause: bool,
) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut search_start = 0;

    while search_start < buffer.len() {
        let allow_clause = allow_first_clause && !*first_chunk_sent;
        let Some(boundary) = find_boundary(buffer, search_start, allow_clause) else {
            break;
        };

        let chunk = buffer
            .get(..boundary.final_end_pos)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        buffer.drain(..boundary.remainder_offset.min(buffer.len()));
        search_start = 0;

        if !chunk.is_empty() {
            *first_chunk_sent = true;
            chunks.push(chunk);
        }
    }

    chunks
}

/// Extracts speech chunks from an accumulator buffer.
///
/// If `first_chunk_sent` is false, this extracts the first chunk at the first
/// clause boundary (',', ';', ':', '—') once the accumulated text reaches
/// [`MIN_FIRST_CLAUSE_CHARS`], or at the first terminal sentence boundary
/// ('.', '!', '?', '\n').
///
/// Once the first chunk has been emitted (`*first_chunk_sent == true`), all
/// subsequent chunks maintain full sentence granularity.
///
/// Respects decimals (e.g. "3,5"), abbreviations ("Dr.", "Sr."), and ellipses ("...").
/// Drains emitted chunks from `buffer` and leaves trailing incomplete text.
#[must_use]
pub fn extract_completed_chunks(buffer: &mut String, first_chunk_sent: &mut bool) -> Vec<String> {
    extract_completed_chunks_inner(buffer, first_chunk_sent, true)
}

/// Extracts completed sentences from an accumulator buffer without clause splitting.
///
/// Looks for terminal punctuation ('.', '!', '?', or '\n') followed by whitespace,
/// while respecting ellipses ('...'), abbreviations ("Dr."), and trailing quotes.
/// Drains completed sentences from `buffer` and leaves trailing incomplete text.
#[must_use]
pub fn extract_completed_sentences(buffer: &mut String) -> Vec<String> {
    let mut dummy_first = true;
    extract_completed_chunks_inner(buffer, &mut dummy_first, false)
}

/// Splits an entire text reply into synthesis chunks.
///
/// Emits the first clause early if it reaches [`MIN_FIRST_CLAUSE_CHARS`],
/// and splits all subsequent text by sentence boundaries.
#[must_use]
pub fn extract_reply_chunks(text: &str) -> Vec<String> {
    let mut buf = text.to_string();
    let mut first_sent = false;
    let mut chunks = extract_completed_chunks(&mut buf, &mut first_sent);
    let leftover = buf.trim();
    if !leftover.is_empty() {
        chunks.push(leftover.to_string());
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_completed_sentences_various_cases() {
        let mut buffer = "E aí, Gabriel! Beleza? Tô de boa por aqui.".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(
            sentences,
            vec!["E aí, Gabriel!", "Beleza?", "Tô de boa por aqui."]
        );
        assert!(buffer.is_empty());

        let mut buffer = "Opa, peraí... Ainda não acabei!".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(sentences, vec!["Opa, peraí...", "Ainda não acabei!"]);
        assert!(buffer.is_empty());

        let mut buffer = "O Dr. House é médico. Dr. House não atende hoje.".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(
            sentences,
            vec!["O Dr. House é médico.", "Dr. House não atende hoje."]
        );
        assert!(buffer.is_empty());

        // Incomplete sentence left in buffer
        let mut buffer = "Frase completa. Frase incomp".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(sentences, vec!["Frase completa."]);
        assert_eq!(buffer, "Frase incomp");

        // Trailing quote
        let mut buffer = "\"Isso é ótimo!\", disse Gabriel.".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(sentences, vec!["\"Isso é ótimo!\", disse Gabriel."]);
        assert!(buffer.is_empty());

        // Multi-line sentences
        let mut buffer = "Linha um.\nLinha dois!\n".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(sentences, vec!["Linha um.", "Linha dois!"]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn property_chunks_concatenate_to_original_text_without_empty_chunks() {
        let samples = [
            "",
            "   ",
            "Olá!",
            "Oi, tudo bem?",
            "Sim. Vamos lá.",
            "Com certeza Gabriel, eu vou te ajudar hoje! Vamos começar.",
            "O processador opera a 3,5 GHz, o que garante 3.14 vezes mais velocidade.",
            "O Dr. Gabriel ligou; ele volta amanhã com certeza. Até mais.",
            "Pensando bem... tudo vai dar certo no final! Com certeza.",
            "Esta é a primeira cláusula longa — e esta continua depois do travessão. Fim.",
            "Aqui está o relatório: todos os dados foram conferidos com calma.",
            "\"Com certeza Gabriel, podemos fazer isso!\", disse ela com alegria.",
            "Frase um! Frase dois? Frase três. Frase quatro...\nFrase cinco.",
            "Texto longo sem pontuação terminal mas que deve ser retornado intacto no leftover",
        ];

        for text in samples {
            let chunks = extract_reply_chunks(text);

            // Property 1: No chunk is empty
            for chunk in &chunks {
                assert!(
                    !chunk.trim().is_empty(),
                    "chunk must not be empty for input: {text:?}"
                );
            }

            // Property 2: Modulo trimming and spacing, concatenating reproduces the original text
            if text.trim().is_empty() {
                assert!(chunks.is_empty());
            } else {
                let joined = chunks.join(" ");
                let original_words: Vec<&str> = text.split_whitespace().collect();
                let joined_words: Vec<&str> = joined.split_whitespace().collect();
                assert_eq!(
                    joined_words, original_words,
                    "reproduced words must match original for input: {text:?}"
                );
            }
        }
    }

    #[test]
    fn first_clause_splits_at_comma_when_length_reaches_threshold() {
        let text =
            "Com certeza meu caro amigo Gabriel, estou pronto para te ajudar! Vamos em frente.";
        let chunks = extract_reply_chunks(text);

        assert_eq!(
            chunks,
            vec![
                "Com certeza meu caro amigo Gabriel,",
                "estou pronto para te ajudar!",
                "Vamos em frente."
            ]
        );
        assert!(chunks[0].chars().count() >= MIN_FIRST_CLAUSE_CHARS);
    }

    #[test]
    fn first_clause_does_not_split_short_comma_fragment() {
        let text = "Oi, tudo bem com você? Vamos começar agora.";
        let chunks = extract_reply_chunks(text);

        // "Oi," is only 3 chars < MIN_FIRST_CLAUSE_CHARS, so it stays with the first sentence.
        assert_eq!(
            chunks,
            vec!["Oi, tudo bem com você?", "Vamos começar agora."]
        );
    }

    #[test]
    fn later_chunks_maintain_sentence_granularity_even_with_commas() {
        let text =
            "Com certeza Gabriel, estou pronto! Na próxima etapa, faremos o teste, com calma.";
        let chunks = extract_reply_chunks(text);

        // First chunk splits at clause boundary.
        // Later chunks keep sentence granularity despite containing commas.
        assert_eq!(
            chunks,
            vec![
                "Com certeza Gabriel,",
                "estou pronto!",
                "Na próxima etapa, faremos o teste, com calma."
            ]
        );
    }

    #[test]
    fn boundaries_respect_decimals_in_portuguese_and_english() {
        // Portuguese decimal "3,5" uses a comma
        let pt_text = "O sistema consumiu 3,5 watts de energia, o que é excelente.";
        let pt_chunks = extract_reply_chunks(pt_text);
        assert_eq!(
            pt_chunks,
            vec![
                "O sistema consumiu 3,5 watts de energia,",
                "o que é excelente."
            ]
        );

        // English decimal "2.5" uses a dot
        let en_text = "A versão 2.5 foi liberada agora, aproveite para atualizar.";
        let en_chunks = extract_reply_chunks(en_text);
        assert_eq!(
            en_chunks,
            vec![
                "A versão 2.5 foi liberada agora,",
                "aproveite para atualizar."
            ]
        );
    }

    #[test]
    fn boundaries_respect_abbreviations_and_ellipses() {
        let text = "O Dr. Gabriel chegou agora há pouco; vamos conversar com ele.";
        let chunks = extract_reply_chunks(text);
        assert_eq!(
            chunks,
            vec![
                "O Dr. Gabriel chegou agora há pouco;",
                "vamos conversar com ele."
            ]
        );

        let ellipsis_text = "Esperando um pouco... talvez seja melhor agora. Tudo certo.";
        let ellipsis_chunks = extract_reply_chunks(ellipsis_text);
        assert_eq!(
            ellipsis_chunks,
            vec![
                "Esperando um pouco...",
                "talvez seja melhor agora.",
                "Tudo certo."
            ]
        );
    }

    #[test]
    fn boundaries_support_em_dash_colon_and_semicolon() {
        let em_dash = "Com certeza meu caro amigo — vamos resolver essa pendência hoje.";
        let chunks = extract_reply_chunks(em_dash);
        assert_eq!(
            chunks,
            vec![
                "Com certeza meu caro amigo —",
                "vamos resolver essa pendência hoje."
            ]
        );

        let colon = "Aqui está a solução completa: todos os testes passaram.";
        let colon_chunks = extract_reply_chunks(colon);
        assert_eq!(
            colon_chunks,
            vec!["Aqui está a solução completa:", "todos os testes passaram."]
        );
    }
}
