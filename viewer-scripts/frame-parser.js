export const ENVELOPE_HEADER_LEN = 17;
export const ENVELOPE_VERSION_V1 = 1;
export const CONTENT_TYPE_FLOAT32_TENSOR = 1;

export function parseEnvelopeV1(arrayBuffer) {
  if (arrayBuffer.byteLength < ENVELOPE_HEADER_LEN) {
    throw new Error("Frame too small for envelope v1.");
  }

  const view = new DataView(arrayBuffer);
  const version = view.getUint8(0);
  const contentType = view.getUint16(1, true);
  const flags = view.getUint16(3, true);
  const timestampNs = view.getBigUint64(5, true);
  const payloadLen = view.getUint32(13, true);

  if (version !== ENVELOPE_VERSION_V1) {
    throw new Error(`Unsupported envelope version: ${version}`);
  }

  const payloadOffset = ENVELOPE_HEADER_LEN;
  const payloadEnd = payloadOffset + payloadLen;

  if (arrayBuffer.byteLength !== payloadEnd) {
    throw new Error("Envelope payload length mismatch.");
  }

  // Copy payload to new buffer for 4-byte alignment.
  const payloadBuffer = arrayBuffer.slice(payloadOffset, payloadEnd);

  return { contentType, flags, timestampNs, payloadBuffer };
}

export function normalizeSocketPayload(data, errorPrefix) {
  if (data instanceof ArrayBuffer) {
    return data;
  }

  if (data instanceof Blob) {
    throw new Error(
      `${errorPrefix} received a Blob payload unexpectedly; convert it with arrayBuffer() first.`
    );
  }

  throw new Error(`${errorPrefix} payload must be an ArrayBuffer.`);
}

export function parseEnvelopeOrLegacyFloat32(arrayBuffer, legacyValidator, errorPrefix) {
  if (arrayBuffer.byteLength >= ENVELOPE_HEADER_LEN) {
    try {
      return parseEnvelopeV1(arrayBuffer);
    } catch (err) {
      // Fall through to legacy raw Float32 payloads.
    }
  }

  if (arrayBuffer.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`${errorPrefix} payload must be Float32 aligned.`);
  }

  const data = new Float32Array(arrayBuffer);
  legacyValidator(data);

  return {
    contentType: CONTENT_TYPE_FLOAT32_TENSOR,
    flags: 0,
    timestampNs: 0n,
    payloadBuffer: arrayBuffer.slice(0),
  };
}
