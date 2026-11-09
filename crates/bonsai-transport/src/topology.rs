//! Sysfs topology selection using actual CUDA-visible PCI functions.
//! Unlike upstream's global scanner, this never synthesizes ancestor BDFs.
use crate::fabric::err;
use bonsai_core::Result;
use fabric_lib::{DomainInfo, RdmaDomainInfo};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

pub(crate) struct Selection {
    pub domains: Vec<DomainInfo>,
    pub cpus: Vec<u16>,
    pub description: String,
}
pub(crate) fn parse_cpus(list: &str) -> Result<HashSet<u16>> {
    let mut cpus = HashSet::new();
    for item in list.trim().split(',').filter(|s| !s.is_empty()) {
        let mut bounds = item.split('-');
        let start: u16 = bounds.next().unwrap_or("").parse().map_err(err)?;
        let end: u16 = bounds
            .next()
            .map(|s| s.parse())
            .transpose()
            .map_err(err)?
            .unwrap_or(start);
        if end < start || bounds.next().is_some() {
            return Err(err("invalid CPU list"));
        }
        cpus.extend(start..=end);
    }
    Ok(cpus)
}
pub(crate) fn allowed_cpus() -> Result<HashSet<u16>> {
    let status = std::fs::read_to_string("/proc/thread-self/status").map_err(err)?;
    parse_cpus(
        status
            .lines()
            .find_map(|s| s.strip_prefix("Cpus_allowed_list:"))
            .ok_or_else(|| err("scheduler affinity unavailable"))?,
    )
}
fn numa(path: &Path) -> Option<i32> {
    std::fs::read_to_string(path.join("numa_node"))
        .ok()?
        .trim()
        .parse::<i32>()
        .ok()
        .filter(|n| *n >= 0)
}
fn common_depth(a: &Path, b: &Path) -> usize {
    a.components()
        .zip(b.components())
        .take_while(|(x, y)| x == y)
        .count()
}
fn domain_path(domain: &DomainInfo) -> Result<PathBuf> {
    match domain {
        DomainInfo::Verbs(info) => {
            let base = PathBuf::from("/sys/class/infiniband").join(domain.name().as_ref());
            let state =
                std::fs::read_to_string(base.join(format!("ports/{}/state", info.port_num)))
                    .map_err(err)?;
            if !state.trim().starts_with("4:") {
                return Err(err("RDMA port is not ACTIVE"));
            }
            std::fs::canonicalize(base.join("device")).map_err(err)
        }
        DomainInfo::Efa(info) => {
            // fi_getinfo supplies the PCI identity of this visible EFA domain.
            let pci = unsafe {
                info.fi()
                    .as_ref()
                    .nic
                    .as_ref()
                    .and_then(|n| n.bus_attr.as_ref())
                    .ok_or_else(|| err("EFA domain lacks PCI identity"))?
                    .attr
                    .pci
            };
            std::fs::canonicalize(format!(
                "/sys/bus/pci/devices/{:04x}:{:02x}:{:02x}.{:x}",
                pci.domain_id, pci.bus_id, pci.device_id, pci.function_id
            ))
            .map_err(err)
        }
    }
}
pub(crate) fn select(cuda_device: u8, nets: u8) -> Result<Selection> {
    if nets == 0 {
        return Err(err("requested NIC count must be positive"));
    }
    let prop = cuda_lib::rt::cudaGetDeviceProperties(i32::from(cuda_device)).map_err(err)?;
    let bdf = format!(
        "{:04x}:{:02x}:{:02x}.0",
        prop.pciDomainID, prop.pciBusID, prop.pciDeviceID
    );
    let gpu_path = std::fs::canonicalize(format!("/sys/bus/pci/devices/{bdf}")).map_err(err)?;
    let gpu_numa = numa(&gpu_path);
    let allowed = allowed_cpus()?;
    let local = std::fs::read_to_string(gpu_path.join("local_cpulist"))
        .ok()
        .and_then(|s| parse_cpus(&s).ok())
        .unwrap_or_default();
    let mut cpus: Vec<_> = allowed.intersection(&local).copied().collect();
    if cpus.is_empty() {
        cpus.extend(allowed);
    }
    cpus.sort_unstable();
    if cpus.is_empty() {
        return Err(err("scheduler affinity has no usable CPUs"));
    }
    let mut candidates = Vec::new();
    let mut rejected = Vec::new();
    for domain in fabric_lib::visible_domains() {
        match domain_path(&domain) {
            Ok(path) => {
                let same_numa = gpu_numa.is_some() && numa(&path) == gpu_numa;
                let depth = common_depth(&gpu_path, &path);
                let speed = domain.link_speed();
                candidates.push((
                    depth,
                    same_numa,
                    speed,
                    domain.name().into_owned(),
                    path,
                    domain,
                ));
            }
            Err(e) => rejected.push(format!("{}: {e}", domain.name())),
        }
    }
    // Prefer the deepest actual common PCI ancestor, then GPU NUMA locality,
    // then speed. Name only breaks equally local/equally fast hardware ties.
    candidates.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(b.1.cmp(&a.1))
            .then(b.2.cmp(&a.2))
            .then(a.3.cmp(&b.3))
    });
    if candidates.len() < usize::from(nets) {
        return Err(err(format!(
            "requested {nets} NICs but only {} active visible domains; rejected {rejected:?}",
            candidates.len()
        )));
    }
    let mut description = format!(
        "cuda:{cuda_device} PCI={bdf} path={} NUMA={gpu_numa:?} allowed_cpus={cpus:?}",
        gpu_path.display()
    );
    let domains = candidates
        .into_iter()
        .take(usize::from(nets))
        .map(|(depth, same_numa, speed, name, path, domain)| {
            description.push_str(&format!(
                "; NIC={name} path={} common_depth={depth} same_numa={same_numa} link_bps={speed}",
                path.display()
            ));
            domain
        })
        .collect();
    Ok(Selection {
        domains,
        cpus,
        description,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affinity_ranges_and_pci_ancestors() {
        assert_eq!(parse_cpus("1-3,8\n").unwrap(), HashSet::from([1, 2, 3, 8]));
        assert!(parse_cpus("8-2").is_err());
        assert!(parse_cpus("1-2-3").is_err());
        assert!(parse_cpus("70000").is_err());
        assert_eq!(
            common_depth(
                Path::new("/sys/devices/pci0/bridge/gpu"),
                Path::new("/sys/devices/pci0/bridge/nic")
            ),
            5
        );
    }
}
