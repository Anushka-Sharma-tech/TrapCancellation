import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const email = request.nextUrl.searchParams.get("email");

  if (!email || !email.includes("@")) {
    return NextResponse.json({ error: "Valid email is required" }, { status: 400 });
  }

  try {
    const response = await fetch(`https://api.xposedornot.com/v1/check-email/${encodeURIComponent(email)}`, {
      headers: { Accept: "application/json" },
      cache: "no-store"
    });

    if (response.status === 404) {
      return NextResponse.json({ breached: false, count: 0 });
    }

    const data = await response.json();

    return NextResponse.json({
      breached: true,
      count: Array.isArray(data.breaches) ? data.breaches.length : 1,
      source: "XposedOrNot"
    });
  } catch {
    return NextResponse.json({ error: "Breach service unavailable" }, { status: 502 });
  }
}