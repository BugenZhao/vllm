// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::marker::PhantomData;

use bytes::{Buf as _, BufMut as _};
use serde::Serialize;
use serde::de::DeserializeOwned;
use thiserror_ext::AsReport as _;
use tonic::Status;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};

/// gRPC codec that carries messages as MessagePack with named fields, the
/// encoding the engine-core types already use on the ZMQ wire.
#[derive(Debug)]
pub struct MsgpackCodec<T, U>(PhantomData<(T, U)>);

impl<T, U> Default for MsgpackCodec<T, U> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T, U> Codec for MsgpackCodec<T, U>
where
    T: Serialize + Send + 'static,
    U: DeserializeOwned + Send + 'static,
{
    type Encode = T;
    type Decode = U;
    type Encoder = MsgpackEncoder<T>;
    type Decoder = MsgpackDecoder<U>;

    fn encoder(&mut self) -> Self::Encoder {
        MsgpackEncoder(PhantomData)
    }

    fn decoder(&mut self) -> Self::Decoder {
        MsgpackDecoder(PhantomData)
    }
}

#[derive(Debug)]
pub struct MsgpackEncoder<T>(PhantomData<T>);

impl<T: Serialize> Encoder for MsgpackEncoder<T> {
    type Item = T;
    type Error = Status;

    fn encode(&mut self, item: T, dst: &mut EncodeBuf<'_>) -> Result<(), Status> {
        rmp_serde::encode::write_named(&mut dst.writer(), &item).map_err(|error| {
            Status::internal(format!("failed to encode message: {}", error.as_report()))
        })
    }
}

#[derive(Debug)]
pub struct MsgpackDecoder<U>(PhantomData<U>);

impl<U: DeserializeOwned> Decoder for MsgpackDecoder<U> {
    type Item = U;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<U>, Status> {
        let message = src.copy_to_bytes(src.remaining());
        rmp_serde::from_slice(&message).map(Some).map_err(|error| {
            Status::internal(format!("failed to decode message: {}", error.as_report()))
        })
    }
}
