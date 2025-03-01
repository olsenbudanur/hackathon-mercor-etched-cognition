import { useState, useEffect } from "react";
import "./App.css"; // Import your custom CSS file

export default function StreamingApp() {
  const [words, setWords] = useState([]);
  const [typing, setTyping] = useState(true);

  // Function to determine the color based on the number
  const getColor = (number) => {
    switch (number) {
      case 1:
        return "text-[#26231A]"; // Black-ish
      case 2:
        return "text-[#8F8C84]"; // Gray
      case 3:
        return "text-[#D97757]"; // Orange
      default:
        return "text-black"; // Default color for other numbers
    }
  };

  useEffect(() => {
    console.log("Connecting to SSE...");

    const eventSource = new EventSource("http://127.0.0.1:8000/stream");

    eventSource.onopen = () => console.log("SSE Connection Opened");
    eventSource.onerror = (error) => console.error("SSE Error:", error);
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data); // Parse the incoming data
        setWords((prev) => [...prev, { word: data.word, number: data.number }]);
        setTyping(true); // Trigger typing effect
      } catch (error) {
        console.error("Error parsing JSON:", error);
      }
    };

    return () => {
      console.log("Closing SSE...");
      eventSource.close();
    };
  }, []);

  return (
    <div className="min-h-screen flex justify-center items-center p-4 bg-gradient-to-b from-[#F0EFE7] to-[#F9F8F5] relative">
      {/* Gradient Reference Bar */}
      <div className="absolute top-6 left-6 flex items-center space-x-1.5">
        <div className="w-4 h-24 bg-gradient-to-b from-[#8F8C84] via-[#26231A] to-[#D97757] rounded-lg flex flex-col justify-between">
          {/* Gradient bar */}
        </div>
        <div className="flex flex-col justify-between h-24 text-sm">
          {/* Aligning text with the gradient bar */}
          <span className="text-[#8F8C84] flex-3">Cursory Awareness</span>
          <span></span>
          <span className="text-[#26231A] flex-3">Focused Attention</span>
          <span></span>
          <span className="text-[#D97757] flex-3">Complete Absorption</span>
        </div>
      </div>

      <div className="w-full max-w-4xl bg-white rounded-lg p-6 shadow-lg border border-gray-200">
        <h1 className="text-3xl text-[#26231A] mb-6 text-center">
          Attention really is all you need!
        </h1>
        <div className="message-box">
          <pre className="text-base text-[#8F8C84] whitespace-normal overflow-auto break-words h-96 p-4 bg-[#F6F5F0] rounded-md">
            {words.map((item, index) => (
              <span key={index} className={`${getColor(item.number)} inline`}>
                {item.word}
              </span>
            ))}
            {typing && <span className="typing-effect"></span>}
          </pre>
        </div>
      </div>
    </div>
  );
}
