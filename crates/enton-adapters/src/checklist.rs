//! The owner's checklist: what Enton may bring up on its own.
//!
//! `CHECKLIST.md` lives in Enton's config directory (`$XDG_CONFIG_HOME/enton`, else
//! `~/.config/enton`). The owner writes it by hand, as Markdown; Enton only reads it,
//! at startup and whenever it changes. Only whether it holds something to check enters
//! the core (as [`enton_core::Event::Checklist`]), so a drive with nothing to check
//! spends nothing. The text itself goes to the cortex with a drive thought and never
//! reaches the reducer or the soul.
//!
//! A checklist that is effectively empty holds nothing to check: only blank lines,
//! headings, empty list items (`-`, `- [ ]`, `1.`), thematic breaks (`---`) and
//! one-line HTML comments. That is `OpenClaw`'s rule for its `HEARTBEAT.md`, which lets a
//! template of headings sit in place without buying a model call, plus the two kinds of
//! line that carry no content either.

use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// The checklist's file name inside Enton's config directory.
pub const FILE_NAME: &str = "CHECKLIST.md";

/// The largest checklist Enton reads, in bytes: 4 KiB, about 1,000 tokens, a quarter of
/// the cortex's default context. A larger file is not used (never truncated): nothing
/// to check until it is shortened.
pub const MAX_CHECKLIST_BYTES: u64 = 4 * 1024;

/// Where the checklist lives: `$XDG_CONFIG_HOME/enton/CHECKLIST.md`, falling back to
/// `~/.config/enton/CHECKLIST.md`. `None` when neither variable names a directory.
#[must_use]
pub fn default_path() -> Option<PathBuf> {
    path_from(
        std::env::var_os("XDG_CONFIG_HOME"),
        std::env::var_os("HOME"),
    )
}

/// [`default_path`] for the given `XDG_CONFIG_HOME` and `HOME` values. An empty value
/// counts as unset, as the XDG specification asks.
#[must_use]
pub fn path_from(config_home: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    let set = |value: Option<OsString>| value.filter(|value| !value.is_empty());
    let config = set(config_home)
        .map(PathBuf::from)
        .or_else(|| set(home).map(|home| PathBuf::from(home).join(".config")))?;
    Some(config.join("enton").join(FILE_NAME))
}

/// Whether `text` holds something to check: any line that is not blank, a heading, an
/// empty list item, a thematic break or a one-line HTML comment.
#[must_use]
pub fn is_actionable(text: &str) -> bool {
    text.lines().any(|line| !is_empty_line(line.trim()))
}

/// The lines of `text` that hold something to check, trimmed and without their list
/// marker (`-`, `*`, `+`, `1.` or `1)` followed by a space): what a drive may bring up,
/// one by one. A checkbox stays, so a ticked item still reads as done.
pub fn items(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(str::trim)
        .filter(|line| !is_empty_line(line))
        .map(unmarked)
}

/// A trimmed line without its list marker, when it has one.
fn unmarked(line: &str) -> &str {
    let rest = if let Some(rest) = line.strip_prefix(['-', '*', '+']) {
        Some(rest)
    } else {
        let number = line.trim_start_matches(|c: char| c.is_ascii_digit());
        if number.len() < line.len() {
            number.strip_prefix(['.', ')'])
        } else {
            None
        }
    };
    rest.filter(|rest| rest.starts_with(char::is_whitespace))
        .map_or(line, str::trim_start)
}

/// A trimmed line that carries nothing to check.
fn is_empty_line(line: &str) -> bool {
    line.is_empty()
        || is_heading(line)
        || is_empty_list_item(line)
        || is_thematic_break(line)
        || is_comment(line)
}

/// An ATX heading: one to six `#`, then the end of the line or whitespace.
fn is_heading(line: &str) -> bool {
    let rest = line.trim_start_matches('#');
    let level = line.len() - rest.len();
    (1..=6).contains(&level) && rest.chars().next().is_none_or(char::is_whitespace)
}

/// A list marker (`-`, `*`, `+`, or a number followed by `.` or `)`) with nothing after
/// it but, perhaps, an empty or ticked checkbox.
fn is_empty_list_item(line: &str) -> bool {
    let rest = if let Some(rest) = line.strip_prefix(['-', '*', '+']) {
        rest
    } else {
        let number = line.trim_start_matches(|c: char| c.is_ascii_digit());
        if number.len() == line.len() {
            return false;
        }
        match number.strip_prefix(['.', ')']) {
            Some(rest) => rest,
            None => return false,
        }
    };
    matches!(rest.trim(), "" | "[ ]" | "[]" | "[x]" | "[X]")
}

/// Three or more `-`, `*` or `_`, all the same, with any spaces between them.
fn is_thematic_break(line: &str) -> bool {
    let mut marks = line.chars().filter(|c| !c.is_whitespace());
    let Some(mark) = marks.next().filter(|mark| matches!(mark, '-' | '*' | '_')) else {
        return false;
    };
    let mut count = 1;
    for other in marks {
        if other != mark {
            return false;
        }
        count += 1;
    }
    count >= 3
}

/// A whole line of HTML comment: `<!-- ... -->`.
fn is_comment(line: &str) -> bool {
    line.len() >= 7 && line.starts_with("<!--") && line.ends_with("-->")
}

/// What the checklist file holds, as Enton reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Checklist {
    /// There is no checklist file, or no place to look for one.
    Missing,
    /// The file holds nothing to check (see [`is_actionable`]).
    Empty,
    /// Something to check: the file's text, as the owner wrote it.
    Actionable(String),
    /// The file cannot be used, for the reason given, so there is nothing to check.
    Unusable(String),
}

impl Checklist {
    /// Read the checklist at `path`.
    #[must_use]
    pub fn read(path: &Path) -> Self {
        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Self::Missing,
            Err(err) => return Self::Unusable(format!("cannot read it: {err}")),
        };
        if !metadata.is_file() {
            return Self::Unusable("it is not a regular file".to_owned());
        }
        if metadata.len() > MAX_CHECKLIST_BYTES {
            return Self::too_large(metadata.len());
        }
        let mut bytes = Vec::new();
        let read = fs::File::open(path)
            .and_then(|file| file.take(MAX_CHECKLIST_BYTES + 1).read_to_end(&mut bytes));
        if let Err(err) = read {
            return Self::Unusable(format!("cannot read it: {err}"));
        }
        // It may have grown between the size check and the read.
        if bytes.len() as u64 > MAX_CHECKLIST_BYTES {
            return Self::too_large(bytes.len() as u64);
        }
        match String::from_utf8(bytes) {
            Ok(text) if is_actionable(&text) => Self::Actionable(text),
            Ok(_) => Self::Empty,
            Err(_) => Self::Unusable("it is not UTF-8 text".to_owned()),
        }
    }

    fn too_large(bytes: u64) -> Self {
        Self::Unusable(format!(
            "it is {bytes} bytes, over the {MAX_CHECKLIST_BYTES}-byte cap: shorten it"
        ))
    }

    /// Whether there is something to check.
    #[must_use]
    pub fn is_actionable(&self) -> bool {
        matches!(self, Self::Actionable(_))
    }

    /// The text to check, when there is something to check.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Actionable(text) => Some(text),
            Self::Missing | Self::Empty | Self::Unusable(_) => None,
        }
    }
}

/// What a poll saw of the file: whether it exists, and its size and modification time.
type Stamp = Option<(u64, Option<SystemTime>)>;

/// Watches the checklist file cheaply: each poll reads its metadata, and reads the file
/// again only when that changed.
#[derive(Debug, Clone)]
pub struct ChecklistWatcher {
    path: Option<PathBuf>,
    /// What the last poll saw; `None` before the first.
    seen: Option<Stamp>,
}

impl ChecklistWatcher {
    /// A watcher of the checklist at `path` (`None`: there is nowhere to look).
    #[must_use]
    pub fn new(path: Option<PathBuf>) -> Self {
        Self { path, seen: None }
    }

    /// The file this watcher reads, if any.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// The checklist, when it may have changed: at the first poll, then whenever the
    /// file appears, disappears, or changes size or modification time. `None` when
    /// nothing changed.
    pub fn poll(&mut self) -> Option<Checklist> {
        let stamp = self.path.as_deref().and_then(|path| {
            fs::metadata(path)
                .ok()
                .map(|metadata| (metadata.len(), metadata.modified().ok()))
        });
        if self.seen == Some(stamp) {
            return None;
        }
        self.seen = Some(stamp);
        Some(
            self.path
                .as_deref()
                .map_or(Checklist::Missing, Checklist::read),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headings_blank_lines_and_empty_items_are_nothing_to_check() {
        for empty in [
            "",
            "\n\n   \n",
            "# CHECKLIST.md\n\n## Morning\n\n### Evening\n",
            "#\n######\n",
            "- \n* [ ]\n+ [x]\n- [X]\n-\n1.\n2)\n10. [ ]\n",
            "# Things to bring up\n\n- [ ]\n- [ ]\n",
            "---\n***\n_ _ _\n- - -\n",
            "<!-- Add what Enton may bring up on its own. -->\n",
            "\t# Indented heading\n\t-\n",
        ] {
            assert!(!is_actionable(empty), "{empty:?}");
        }
    }

    #[test]
    fn items_are_the_lines_with_content_without_their_list_markers() {
        let text = "# Hoje\n\n- [ ] regar as plantas\n* ligar para a mae\n  1. tomar o remedio\n\
                    2) [x] pagar a luz\n- \n---\n1.5 kg de farinha\n-sem espaco\n<!-- nota -->\n";
        assert_eq!(
            items(text).collect::<Vec<_>>(),
            [
                "[ ] regar as plantas",
                "ligar para a mae",
                "tomar o remedio",
                "[x] pagar a luz",
                "1.5 kg de farinha",
                "-sem espaco",
            ]
        );
        assert_eq!(items("# only a heading\n- [ ]\n").count(), 0);
    }

    #[test]
    fn any_line_with_content_is_something_to_check() {
        for actionable in [
            "Remind me to water the plants",
            "# Daily\n- [ ] water the plants\n",
            "- regar as plantas",
            "* [ ] ligar para a mae",
            "1. take the medicine",
            "#hashtag",
            "####### seven hashes is not a heading",
            "-- not a break",
            "<!-- unclosed comment",
            "[ ]",
            "1.5 kg de farinha",
        ] {
            assert!(is_actionable(actionable), "{actionable:?}");
        }
    }

    #[test]
    fn the_path_follows_xdg_config_home_then_home() {
        let some = |value: &str| Some(OsString::from(value));
        assert_eq!(
            path_from(some("/xdg"), some("/home/owner")),
            Some(PathBuf::from("/xdg/enton/CHECKLIST.md"))
        );
        assert_eq!(
            path_from(some(""), some("/home/owner")),
            Some(PathBuf::from("/home/owner/.config/enton/CHECKLIST.md"))
        );
        assert_eq!(
            path_from(None, some("/home/owner")),
            Some(PathBuf::from("/home/owner/.config/enton/CHECKLIST.md"))
        );
        assert_eq!(path_from(None, some("")), None);
        assert_eq!(path_from(None, None), None);
    }

    /// A scratch directory under the system temp dir, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("enton-checklist-{name}-{}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            drop(fs::remove_dir_all(&self.0));
        }
    }

    #[test]
    fn reading_tells_missing_empty_actionable_and_unusable_apart() {
        let scratch = Scratch::new("read");
        let path = scratch.0.join(FILE_NAME);
        assert_eq!(Checklist::read(&path), Checklist::Missing);
        fs::write(&path, "# Checklist\n\n- [ ]\n").unwrap();
        assert_eq!(Checklist::read(&path), Checklist::Empty);
        fs::write(&path, "# Checklist\n- [ ] regar as plantas\n").unwrap();
        let read = Checklist::read(&path);
        assert!(read.is_actionable());
        assert_eq!(read.text(), Some("# Checklist\n- [ ] regar as plantas\n"));
        fs::write(&path, "x".repeat(4097)).unwrap();
        assert!(matches!(Checklist::read(&path), Checklist::Unusable(why) if why.contains("cap")));
        fs::write(&path, [0x66, 0xff, 0xfe]).unwrap();
        assert!(
            matches!(Checklist::read(&path), Checklist::Unusable(why) if why.contains("UTF-8"))
        );
        let directory = Checklist::read(&scratch.0);
        assert!(matches!(directory, Checklist::Unusable(why) if why.contains("regular file")));
    }

    #[test]
    fn the_watcher_reports_the_first_poll_and_then_only_changes() {
        let scratch = Scratch::new("watch");
        let path = scratch.0.join(FILE_NAME);
        let mut watcher = ChecklistWatcher::new(Some(path.clone()));
        assert_eq!(watcher.path(), Some(path.as_path()));
        assert_eq!(watcher.poll(), Some(Checklist::Missing));
        assert_eq!(watcher.poll(), None);
        fs::write(&path, "- regar as plantas\n").unwrap();
        assert!(
            watcher
                .poll()
                .is_some_and(|checklist| checklist.is_actionable())
        );
        assert_eq!(watcher.poll(), None);
        // A different size is a change even within the file system's time resolution.
        fs::write(&path, "# only a heading\n").unwrap();
        assert_eq!(watcher.poll(), Some(Checklist::Empty));
        fs::remove_file(&path).unwrap();
        assert_eq!(watcher.poll(), Some(Checklist::Missing));
        assert_eq!(watcher.poll(), None);

        let mut nowhere = ChecklistWatcher::new(None);
        assert_eq!(nowhere.poll(), Some(Checklist::Missing));
        assert_eq!(nowhere.poll(), None);
    }
}
