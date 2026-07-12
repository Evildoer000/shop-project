import { useEffect, useState } from "react";

export function createId(prefix: string): string {
  const suffix = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID().slice(0, 8)
    : Math.random().toString(16).slice(2, 10);
  return `${prefix}_${suffix}`;
}

function readStorage<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeStorage<T>(key: string, value: T) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore persistence failures
  }
}

export function usePersistentState<T>(key: string, fallback: T) {
  const [value, setValue] = useState<T>(() => readStorage(key, fallback));

  useEffect(() => {
    writeStorage(key, value);
  }, [key, value]);

  return [value, setValue] as const;
}
