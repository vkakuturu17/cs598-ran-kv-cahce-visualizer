import json
import threading
import time
from pathlib import Path
from typing import Any, Tuple

import msgspec
import zmq
from msgspec.msgpack import Decoder

from vllm.v1.core.kv_cache_utils import ExternalBlockHash


class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
	ts: float
	events: list[Any]
	data_parallel_rank: int | None = None


class KVCacheEvent(
	msgspec.Struct, array_like=True, omit_defaults=True, gc=False, tag=True
):
	pass


class BlockStored(KVCacheEvent):
	block_hashes: list[ExternalBlockHash]
	parent_block_hash: ExternalBlockHash | None
	token_ids: list[int]
	block_size: int
	lora_id: int | None
	medium: str | None
	lora_name: str | None
	extra_keys: list[Tuple[Any, ...] | None] | None = None
	group_idx: int | None = None


class BlockRemoved(KVCacheEvent):
	block_hashes: list[ExternalBlockHash]
	medium: str | None
	group_idx: int | None = None


class AllBlocksCleared(KVCacheEvent):
	pass


class KVEventBatch(EventBatch):
	events: list[BlockStored | BlockRemoved | AllBlocksCleared]


def _hash_to_key(block_hash: ExternalBlockHash) -> str:
	if isinstance(block_hash, tuple):
		return "|".join(str(part) for part in block_hash)
	return str(block_hash)


def _event_to_dict(event: KVCacheEvent) -> dict[str, Any]:
	if isinstance(event, BlockStored):
		return {
			"type": "BlockStored",
			"block_hashes": [_hash_to_key(h) for h in event.block_hashes],
			"parent_block_hash": (
				_hash_to_key(event.parent_block_hash)
				if event.parent_block_hash is not None
				else None
			),
			"token_ids": list(event.token_ids),
			"block_size": event.block_size,
			"lora_id": event.lora_id,
			"lora_name": event.lora_name,
			"medium": event.medium,
			"group_idx": event.group_idx,
		}

	if isinstance(event, BlockRemoved):
		return {
			"type": "BlockRemoved",
			"block_hashes": [_hash_to_key(h) for h in event.block_hashes],
			"medium": event.medium,
			"group_idx": event.group_idx,
		}

	return {"type": "AllBlocksCleared"}


class KVEventCollector:
	def __init__(
		self,
		endpoint: str = "tcp://127.0.0.1:5557",
		topic: str = "kv-events",
		poll_timeout_ms: int = 100,
	):
		self.endpoint = endpoint
		self.topic = topic
		self.poll_timeout_ms = poll_timeout_ms
		self.decoder = Decoder(type=KVEventBatch)

		self._ctx: zmq.Context | None = None
		self._sub: zmq.Socket | None = None
		self._thread: threading.Thread | None = None
		self._stop_event = threading.Event()

		self.events: list[dict[str, Any]] = []
		self._event_index = 0

	def start(self) -> None:
		if self._thread is not None:
			return

		self._ctx = zmq.Context.instance()
		self._sub = self._ctx.socket(zmq.SUB)
		self._sub.connect(self.endpoint)
		self._sub.setsockopt_string(zmq.SUBSCRIBE, self.topic)

		self._thread = threading.Thread(target=self._run, daemon=True)
		self._thread.start()

	def stop(self) -> None:
		self._stop_event.set()
		if self._thread is not None:
			self._thread.join(timeout=2.0)
			self._thread = None
		if self._sub is not None:
			self._sub.close(0)
			self._sub = None

	def _run(self) -> None:
		assert self._sub is not None
		while not self._stop_event.is_set():
			if self._sub.poll(self.poll_timeout_ms):
				parts = self._sub.recv_multipart()
				if len(parts) != 3:
					continue

				_, seq_bytes, payload = parts
				seq = int.from_bytes(seq_bytes, "big", signed=True)
				batch = self.decoder.decode(payload)

				for event in batch.events:
					self.events.append(
						{
							"event_index": self._event_index,
							"sequence": seq,
							"timestamp": batch.ts,
							"data_parallel_rank": batch.data_parallel_rank,
							"event": _event_to_dict(event),
						}
					)
					self._event_index += 1

	def wait_until_events(self, min_events: int, timeout_s: float = 10.0) -> bool:
		deadline = time.time() + timeout_s
		while time.time() < deadline:
			if len(self.events) >= min_events:
				return True
			time.sleep(0.05)
		return len(self.events) >= min_events

	def write_jsonl(self, output_path: str | Path) -> None:
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w", encoding="utf-8") as f:
			for item in self.events:
				f.write(json.dumps(item) + "\n")