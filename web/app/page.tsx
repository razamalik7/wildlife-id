"use client";  

export default function Home() {
  async function pingBackend() {
    try {
      const res = await fetch("http://127.0.0.1:8000/health");
      const data = await res.json();
      alert(JSON.stringify(data));
    } catch (err) {
      alert("Error contacting backend");
    }
  }

  return (
    <main style={{ padding: 20 }}>
      <h1>WildlifeID</h1>
      <button onClick={pingBackend}>
        Ping backend
      </button>
    </main>
  );
}