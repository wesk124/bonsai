//! Where finished `HandoffReport`s go. JSONL keeps analysis in pandas trivial.

use std::{fs::File, io::{BufWriter, Write}, path::Path};

use bonsai_core::HandoffReport;

pub trait MetricsSink: Send {
    fn record(&mut self, report: &HandoffReport);
    fn flush(&mut self) {}
}

pub struct JsonlSink {
    out: BufWriter<File>,
}

impl JsonlSink {
    pub fn create(path: &Path) -> std::io::Result<Self> {
        Ok(JsonlSink { out: BufWriter::new(File::create(path)?) })
    }
}

impl MetricsSink for JsonlSink {
    fn record(&mut self, report: &HandoffReport) {
        if let Ok(line) = serde_json::to_string(report) {
            let _ = writeln!(self.out, "{line}");
        }
    }

    fn flush(&mut self) {
        let _ = self.out.flush();
    }
}

#[derive(Default)]
pub struct MemorySink {
    pub reports: Vec<HandoffReport>,
}

impl MetricsSink for MemorySink {
    fn record(&mut self, report: &HandoffReport) {
        self.reports.push(report.clone());
    }
}
