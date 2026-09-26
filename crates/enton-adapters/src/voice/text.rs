//! What the synthesizer is given to say: a reply without the stage directions and
//! markup a language model sometimes writes, which a voice would read out loud.

/// Sentence-final punctuation.
const TERMINATORS: [char; 4] = ['.', '!', '?', '…'];

/// Punctuation that takes no space before it.
const CLOSING: [char; 8] = ['.', ',', '!', '?', ';', ':', '…', ')'];

/// The longest parenthetical, in words, that can be a stage direction such as
/// "(risos)" or "(pausa dramática)"; a longer one is running text.
const STAGE_DIRECTION_WORDS: usize = 4;

/// `text` as it should be spoken: without `*actions*`, `[tags]`, stage directions in
/// parentheses such as "(risos)" that stand as a sentence of their own, Markdown
/// emphasis and code markers (their words stay), list and heading markers, and emojis;
/// with runs of whitespace collapsed and no space left before punctuation.
///
/// Numbers, punctuation and parentheses inside running text stay. A single-asterisk
/// span is taken as an action (the chat convention for "*sorri*"), so its words go;
/// `**bold**`, `__bold__` and `_italic_` keep theirs. Empty when nothing speakable is
/// left, such as a reply that was only "*risos*" or an emoji.
#[must_use]
pub fn speakable(text: &str) -> String {
    let chars: Vec<char> = text.chars().filter(|c| !is_emoji(*c)).collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        i = match c {
            '[' => tag(&chars, i, &mut out),
            '*' => asterisks(&chars, i, &mut out),
            '(' => parenthetical(&chars, i, &mut out),
            '_' => underscore(&chars, i, &mut out),
            '`' => i + 1,
            '#' | '-' if at_line_start(&out) && next_is_space(&chars, i) => {
                marker_end(&chars, i, c)
            }
            _ => {
                out.push(c);
                i + 1
            }
        };
    }
    tidy(&out)
}

/// Whether `c` is an emoji or a part of one (a variation selector, a zero-width
/// joiner, a skin tone, a keycap or a flag letter).
fn is_emoji(c: char) -> bool {
    matches!(
        u32::from(c),
        0x1F000..=0x1FAFF // pictographs, emoticons, flags, skin tones
            | 0x2600..=0x27BF // miscellaneous symbols and dingbats
            | 0x2300..=0x23FF // watches, hourglasses, media controls
            | 0x2B00..=0x2BFF // stars and squares
            | 0xFE00..=0xFE0F // variation selectors
            | 0x200D // zero-width joiner
            | 0x20E3 // combining keycap
            | 0xE0000..=0xE007F // tag characters of subdivision flags
    )
}

/// Index of the first `close` at or after `from`, on the same line.
fn find_close(chars: &[char], from: usize, close: char) -> Option<usize> {
    chars
        .iter()
        .skip(from)
        .take_while(|c| **c != '\n')
        .position(|c| *c == close)
        .map(|offset| from + offset)
}

/// Whether `out` so far ends a sentence: it is empty, or its last visible character
/// ends one, or a line break follows it.
fn at_sentence_start(out: &str) -> bool {
    let visible = out.trim_end();
    visible.is_empty() || out[visible.len()..].contains('\n') || visible.ends_with(TERMINATORS)
}

/// Whether `out` so far is at the start of a line, ignoring spaces.
fn at_line_start(out: &str) -> bool {
    let visible = out.trim_end_matches([' ', '\t']);
    visible.is_empty() || visible.ends_with('\n')
}

fn next_is_space(chars: &[char], i: usize) -> bool {
    chars.get(i + 1).is_some_and(|c| *c == ' ')
}

/// Past a heading's `#` run or a list's `-`, and the space after it.
fn marker_end(chars: &[char], i: usize, marker: char) -> usize {
    let mut end = i;
    while chars.get(end) == Some(&marker) {
        end += 1;
    }
    end + usize::from(chars.get(end) == Some(&' '))
}

/// After an element removed as a sentence of its own, skip the punctuation that ended
/// that sentence, so "(risos). Tá bom." leaves "Tá bom." rather than ". Tá bom.".
fn skip_own_terminator(chars: &[char], mut i: usize, whole_sentence: bool) -> usize {
    if whole_sentence {
        while chars.get(i).is_some_and(|c| TERMINATORS.contains(c)) {
            i += 1;
        }
    }
    i
}

/// `[tag]` goes; a Markdown link `[text](target)` keeps its text. An unclosed `[`
/// is dropped.
fn tag(chars: &[char], i: usize, out: &mut String) -> usize {
    let Some(close) = find_close(chars, i + 1, ']') else {
        return i + 1;
    };
    if chars.get(close + 1) == Some(&'(')
        && let Some(target_end) = find_close(chars, close + 2, ')')
    {
        out.extend(chars.get(i + 1..close).unwrap_or_default());
        return target_end + 1;
    }
    skip_own_terminator(chars, close + 1, at_sentence_start(out))
}

/// `**bold**` keeps its words; `*action*` goes; a lone `*` (a bullet, or half of a
/// span split across sentences) is dropped. An asterisk between spaces or digits, as
/// in "2 * 3", is arithmetic and stays.
fn asterisks(chars: &[char], i: usize, out: &mut String) -> usize {
    let before = i.checked_sub(1).and_then(|at| chars.get(at));
    let after = chars.get(i + 1);
    let opens = after.is_some_and(|c| !c.is_whitespace() && *c != '*')
        && !before.is_some_and(|c| c.is_alphanumeric());
    if after == Some(&'*') {
        // A `**` pair: the markers go, the words stay.
        return i + 2;
    }
    if !opens {
        if before.is_some_and(|c| c.is_whitespace() || c.is_ascii_digit())
            && after.is_some_and(|c| c.is_whitespace() || c.is_ascii_digit())
        {
            out.push('*');
        }
        return i + 1;
    }
    match find_close(chars, i + 1, '*') {
        Some(close) => skip_own_terminator(chars, close + 1, at_sentence_start(out)),
        None => i + 1,
    }
}

/// A parenthetical that stands as a sentence of its own and reads as a stage
/// direction ("(risos)", "(suspira fundo)") goes; any other stays, as running text.
fn parenthetical(chars: &[char], i: usize, out: &mut String) -> usize {
    let whole = find_close(chars, i + 1, ')').filter(|close| {
        let inside = chars.get(i + 1..*close).unwrap_or_default();
        at_sentence_start(out) && is_stage_direction(inside) && ends_sentence(chars, close + 1)
    });
    match whole {
        Some(close) => skip_own_terminator(chars, close + 1, true),
        None => {
            out.push('(');
            i + 1
        }
    }
}

/// A few words of letters: no digits and no punctuation of its own.
fn is_stage_direction(inside: &[char]) -> bool {
    let words = inside
        .split(|c| c.is_whitespace())
        .filter(|word| !word.is_empty())
        .count();
    (1..=STAGE_DIRECTION_WORDS).contains(&words)
        && inside
            .iter()
            .all(|c| c.is_alphabetic() || c.is_whitespace() || matches!(c, '-' | '\''))
}

/// Whether what follows index `from` starts a new sentence: nothing, punctuation that
/// ends one, a line break, or a capital (or another stage direction) after a space.
fn ends_sentence(chars: &[char], from: usize) -> bool {
    let rest = chars.get(from..).unwrap_or_default();
    let spaces = rest
        .iter()
        .take_while(|c| **c == ' ' || **c == '\t')
        .count();
    match rest.get(spaces) {
        None | Some('\n') => true,
        Some(c) if TERMINATORS.contains(c) => true,
        Some(c) => spaces > 0 && (c.is_uppercase() || matches!(c, '(' | '*' | '[')),
    }
}

/// `_italic_` and `__bold__` lose their markers; an underscore inside a word, as in
/// "nome_do_arquivo", stays.
fn underscore(chars: &[char], i: usize, out: &mut String) -> usize {
    let inside_word = |at: Option<usize>| {
        at.and_then(|at| chars.get(at))
            .is_some_and(|c| c.is_alphanumeric())
    };
    if inside_word(i.checked_sub(1)) && inside_word(Some(i + 1)) {
        out.push('_');
    }
    i + 1
}

/// Collapse whitespace, drop spaces before closing punctuation and any punctuation left
/// at the start, and give up when nothing speakable remains.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for word in text.split_whitespace() {
        if !out.is_empty() && !word.starts_with(CLOSING) {
            out.push(' ');
        }
        out.push_str(word);
    }
    let out = out.trim_start_matches(|c: char| CLOSING.contains(&c) || c.is_whitespace());
    if out.chars().any(char::is_alphanumeric) {
        out.to_owned()
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::speakable;

    #[test]
    fn stage_directions_and_markup_are_not_spoken() {
        for (reply, spoken) in [
            // Actions between asterisks, wherever they are.
            ("*risos* Tá bom, tá bom.", "Tá bom, tá bom."),
            ("Tá bom. *pisca o olho*", "Tá bom."),
            ("Beleza *risos*.", "Beleza."),
            ("*suspira*. Tá, vou fazer.", "Tá, vou fazer."),
            ("*risos*", ""),
            // Parenthetical stage directions that stand as a sentence of their own.
            ("(risos) Claro que sim!", "Claro que sim!"),
            ("Que dia! (suspira fundo)", "Que dia!"),
            ("(Risos). Brincadeira, mano.", "Brincadeira, mano."),
            ("(risos)", ""),
            // Tags, with or without their own period.
            ("[risos] Tô aqui.", "Tô aqui."),
            ("Tô aqui [pausa] pensando.", "Tô aqui pensando."),
            ("Pronto. [música].", "Pronto."),
            (
                "Veja [o site](https://exemplo.com) depois.",
                "Veja o site depois.",
            ),
            // Markdown emphasis and code keep their words.
            ("Isso é **muito** legal!", "Isso é muito legal!"),
            ("Isso é __muito__ legal!", "Isso é muito legal!"),
            ("_Sério_ mesmo?", "Sério mesmo?"),
            ("Rode `cargo test` agora.", "Rode cargo test agora."),
            ("```rodou```", "rodou"),
            (
                "# Lembrete\nRegar as plantas.",
                "Lembrete Regar as plantas.",
            ),
            (
                "- Primeiro: café.\n- Depois: água.",
                "Primeiro: café. Depois: água.",
            ),
            // Emojis, alone, joined, with skin tones or variation selectors.
            ("Bom dia! 😄☀️", "Bom dia!"),
            ("Valeu 👍🏽, mano!", "Valeu, mano!"),
            ("Família 👨‍👩‍👧 reunida.", "Família reunida."),
            ("😂😂😂", ""),
            // Whitespace and the punctuation a removal leaves behind.
            ("Ok...   beleza  ,  mano\n\n", "Ok... beleza, mano"),
            ("…", ""),
        ] {
            assert_eq!(speakable(reply), spoken, "{reply:?}");
        }
    }

    #[test]
    fn running_text_numbers_and_punctuation_are_kept() {
        for reply in [
            "Oi, Gabriel!",
            "Custa uns 10 reais (mais ou menos).",
            "O Enton (que mora aqui) é legal.",
            "São 3,5 graus (bem frio) às 7h.",
            "Então... tá.",
            "Dr. Silva chega às 14:30, né?",
            "2 * 3 = 6",
            "Salva em nome_do_arquivo.txt, beleza?",
            "A reunião é em 2026-09-26 - não esquece!",
            "Ele disse: \"já volto\".",
            "(2 minutos depois) Voltei!",
        ] {
            assert_eq!(speakable(reply), reply, "{reply:?}");
        }
        // A parenthetical that runs into the sentence is running text.
        assert_eq!(
            speakable("(Na real) eu acho que sim."),
            "(Na real) eu acho que sim."
        );
        // A parenthetical with sentences of its own is content, not a direction.
        assert_eq!(
            speakable("(Aliás, o Gabriel pediu isso ontem.) Pronto."),
            "(Aliás, o Gabriel pediu isso ontem.) Pronto."
        );
    }

    #[test]
    fn unbalanced_markers_are_dropped_not_read_out() {
        // Without its other half a marker proves nothing about the words: they stay.
        for (reply, spoken) in [
            ("*sorri e diz: oi!", "sorri e diz: oi!"),
            ("tchau!*", "tchau!"),
            ("Olha [isso", "Olha isso"),
            ("Um ( solto", "Um ( solto"),
            ("* item solto", "item solto"),
        ] {
            assert_eq!(speakable(reply), spoken, "{reply:?}");
        }
    }
}
