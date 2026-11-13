import { type NextRequest, NextResponse } from "next/server";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

async function proxyRequest(
  request: NextRequest,
  context: RouteContext,
): Promise<Response> {
  const backendBase =
    process.env.MEMORYMATE_BACKEND_URL ??
    (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : undefined);

  if (!backendBase) {
    return NextResponse.json(
      { detail: "The MemoryMate backend service binding is unavailable." },
      { status: 500 },
    );
  }

  const { path } = await context.params;
  const baseUrl = backendBase.endsWith("/") ? backendBase : `${backendBase}/`;
  const targetUrl = new URL(
    path.map((segment) => encodeURIComponent(segment)).join("/"),
    baseUrl,
  );
  targetUrl.search = request.nextUrl.search;

  const headers = new Headers();
  for (const name of ["accept", "content-type", "authorization"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }

  const body =
    request.method === "GET" || request.method === "HEAD"
      ? undefined
      : await request.arrayBuffer();

  const backendResponse = await fetch(targetUrl, {
    method: request.method,
    headers,
    body,
    cache: "no-store",
  });

  const responseHeaders = new Headers();
  for (const name of ["content-type", "cache-control"]) {
    const value = backendResponse.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }

  return new Response(backendResponse.body, {
    status: backendResponse.status,
    headers: responseHeaders,
  });
}

export const GET = proxyRequest;
export const POST = proxyRequest;
export const PATCH = proxyRequest;
export const DELETE = proxyRequest;
