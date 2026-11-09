use bonsai_codec::reference::{HostKv, compare, decode_chunk, encode_chunk};
use bonsai_core::{
    ChunkPlan, ChunkSpec, CodecSpec, ElemType, GroupAxis, KvLayout, QuantBits, QuantParams,
};
use half::{bf16, f16};
fn layout(l: u16, h: u16, d: u16) -> KvLayout {
    KvLayout {
        num_layers: l,
        num_kv_heads: h,
        head_dim: d,
        dtype: ElemType::Bf16,
    }
}
fn q(bits: QuantBits, axis: GroupAxis, symmetric: bool, g: u16) -> CodecSpec {
    let p = QuantParams {
        group_size: g,
        axis,
        symmetric,
    };
    CodecSpec::Quant {
        bits,
        key: p,
        value: p,
    }
}
fn filled(l: KvLayout, t: u32) -> HostKv {
    let mut x = HostKv::zeros(l, t);
    for (li, layer) in x.k.iter_mut().chain(&mut x.v).enumerate() {
        for (i, x) in layer.iter_mut().enumerate() {
            *x = bf16::from_f32((((i + li * 7) % 41) as f32 - 20.0) * 0.125).to_bits();
        }
    }
    x
}
#[test]
fn bf16_rectangles_and_tail_preserve_bits() {
    let l = layout(3, 2, 5);
    let mut src = filled(l, 7);
    src.k[0][0] = 0x7fc1;
    src.v[2][69] = 0xff80;
    let p = ChunkPlan::build(
        l,
        CodecSpec::Bf16,
        7,
        ChunkSpec {
            tokens_per_chunk: 3,
            layers_per_chunk: 2,
        },
    )
    .unwrap();
    let mut dst = HostKv::zeros(l, 7);
    for c in &p.chunks {
        let mut frame = vec![0xcc; c.frame_len as usize];
        encode_chunk(&c.codec, &src, c, &mut frame).unwrap();
        decode_chunk(&c.codec, &frame, c, &mut dst).unwrap();
    }
    assert_eq!(src.k, dst.k);
    assert_eq!(src.v, dst.v);
}
#[test]
fn golden_signed_nibbles_and_rounding() {
    let l = layout(1, 1, 6);
    let spec = q(QuantBits::Int4Packed, GroupAxis::HeadDim, true, 6);
    let mut src = HostKv::zeros(l, 1);
    src.k[0] = [-7.0, -3.5, 0.0, 0.5, 1.5, 7.0]
        .map(|x| bf16::from_f32(x).to_bits())
        .to_vec();
    src.v = src.k.clone();
    let c = &ChunkPlan::build(l, spec, 1, ChunkSpec::WHOLE)
        .unwrap()
        .chunks[0];
    let mut frame = vec![0; c.frame_len as usize];
    encode_chunk(&spec, &src, c, &mut frame).unwrap();
    let at = c.sections.k_payload.offset as usize;
    assert_eq!(&frame[at..at + 3], &[0xc9, 0x00, 0x72]);
    let m = c.sections.k_meta.offset as usize;
    assert_eq!(&frame[m..m + 2], &f16::ONE.to_bits().to_le_bytes());
    let mut dst = HostKv::zeros(l, 1);
    decode_chunk(&spec, &frame, c, &mut dst).unwrap();
    assert_eq!(
        dst.k[0],
        [-7.0, -4.0, 0.0, 0.0, 2.0, 7.0].map(|x| bf16::from_f32(x).to_bits())
    );
}
#[test]
fn all_modes_short_head_and_token_groups() {
    let l = layout(3, 3, 5);
    let src = filled(l, 9);
    for bits in [QuantBits::Int8, QuantBits::Int4Packed] {
        for axis in [GroupAxis::HeadDim, GroupAxis::Token] {
            for symmetric in [true, false] {
                let spec = q(bits, axis, symmetric, 4);
                let p = ChunkPlan::build(
                    l,
                    spec,
                    9,
                    ChunkSpec {
                        tokens_per_chunk: 4,
                        layers_per_chunk: 2,
                    },
                )
                .unwrap();
                let mut dst = HostKv::zeros(l, 9);
                for c in &p.chunks {
                    let mut frame = vec![0xff; c.frame_len as usize];
                    encode_chunk(&spec, &src, c, &mut frame).unwrap();
                    decode_chunk(&spec, &frame, c, &mut dst).unwrap();
                    let n = l.elems(
                        (c.layers.end - c.layers.start) as u64,
                        (c.tokens.end - c.tokens.start) as u64,
                    ) as usize;
                    assert_eq!(
                        c.sections.k_payload.len as usize,
                        if bits == QuantBits::Int8 {
                            n
                        } else {
                            n.div_ceil(2)
                        }
                    );
                    if bits == QuantBits::Int4Packed && n % 2 == 1 {
                        assert_eq!(
                            frame[(c.sections.k_payload.offset + c.sections.k_payload.len - 1)
                                as usize]
                                & 0xf0,
                            0
                        );
                    }
                }
                assert!(compare(&src, &dst).max_abs < 0.4, "{spec:?}");
            }
        }
    }
}
#[test]
fn constant_zero_underflow_and_metadata_rounding() {
    let l = layout(1, 1, 3);
    for symmetric in [true, false] {
        let spec = q(QuantBits::Int4Packed, GroupAxis::HeadDim, symmetric, 3);
        for x in [0.0, 1.234375, 1e-12] {
            let mut src = HostKv::zeros(l, 2);
            src.k[0].fill(bf16::from_f32(x).to_bits());
            src.v = src.k.clone();
            let c = &ChunkPlan::build(l, spec, 2, ChunkSpec::WHOLE)
                .unwrap()
                .chunks[0];
            let mut frame = vec![0; c.frame_len as usize];
            encode_chunk(&spec, &src, c, &mut frame).unwrap();
            let mut dst = HostKv::zeros(l, 2);
            decode_chunk(&spec, &frame, c, &mut dst).unwrap();
            assert!(dst.k[0].iter().all(|&x| bf16::from_bits(x).is_finite()));
            if !symmetric || x == 0.0 {
                let expected =
                    bf16::from_f32(f16::from_f32(bf16::from_f32(x).to_f32()).to_f32()).to_bits();
                assert!(dst.k[0].iter().all(|&x| x == expected));
            }
        }
    }
}
#[test]
fn rejects_bad_buffers_nonfinite_and_metadata() {
    let l = layout(1, 1, 3);
    let spec = q(QuantBits::Int8, GroupAxis::HeadDim, false, 2);
    let c = &ChunkPlan::build(l, spec, 2, ChunkSpec::WHOLE)
        .unwrap()
        .chunks[0];
    let mut src = filled(l, 2);
    let mut f = vec![0; c.frame_len as usize];
    assert!(encode_chunk(&spec, &src, c, &mut f[..1]).is_err());
    for x in [f32::NAN, f32::INFINITY, 1e8] {
        src.k[0][0] = bf16::from_f32(x).to_bits();
        assert!(encode_chunk(&spec, &src, c, &mut f).is_err());
    }
    src = filled(l, 2);
    encode_chunk(&spec, &src, c, &mut f).unwrap();
    let at = c.sections.k_meta.offset as usize;
    f[at..at + 2].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
    let mut dst = HostKv::zeros(l, 2);
    assert!(decode_chunk(&spec, &f, c, &mut dst).is_err());
    assert!(dst.k[0].iter().all(|&x| x == 0));
    let mut bad = c.clone();
    bad.sections.v_payload = bad.sections.k_payload;
    assert!(encode_chunk(&spec, &src, &bad, &mut f).is_err());
}
#[test]
fn compare_reports_known_noise() {
    let l = layout(1, 1, 1);
    let mut a = HostKv::zeros(l, 1);
    a.k[0][0] = bf16::from_f32(2.0).to_bits();
    let b = HostKv::zeros(l, 1);
    let s = compare(&a, &b);
    assert_eq!(s.max_abs, 2.0);
    assert_eq!(s.mse, 2.0);
    assert_eq!(s.sqnr_db, 0.0);
    assert!(compare(&a, &a).sqnr_db.is_infinite());
}

#[test]
fn token_axis_metadata_has_group_major_channel_order() {
    let l = layout(1, 1, 3);
    let spec = q(QuantBits::Int8, GroupAxis::Token, false, 2);
    let mut src = HostKv::zeros(l, 5);
    src.k[0] = [
        -1., 10., 100., 0., 12., 104., 1., 14., 108., 2., 16., 112., 3., 18., 116.,
    ]
    .map(|x| bf16::from_f32(x).to_bits())
    .to_vec();
    src.v = src.k.clone();
    let c = &ChunkPlan::build(l, spec, 5, ChunkSpec::WHOLE)
        .unwrap()
        .chunks[0];
    let mut frame = vec![0; c.frame_len as usize];
    encode_chunk(&spec, &src, c, &mut frame).unwrap();
    let payload = c.sections.k_payload.offset as usize;
    assert_eq!(
        &frame[payload..payload + 15],
        &[0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0]
    );
    let expected = [
        (1. / 255., -1.),
        (2. / 255., 10.),
        (4. / 255., 100.),
        (1. / 255., 1.),
        (2. / 255., 14.),
        (4. / 255., 108.),
        (0., 3.),
        (0., 18.),
        (0., 116.),
    ];
    for (g, (scale, offset)) in expected.into_iter().enumerate() {
        let at = c.sections.k_meta.offset as usize + g * 4;
        assert_eq!(
            &frame[at..at + 2],
            &f16::from_f32(scale).to_bits().to_le_bytes()
        );
        assert_eq!(
            &frame[at + 2..at + 4],
            &f16::from_f32(offset).to_bits().to_le_bytes()
        );
    }
}
