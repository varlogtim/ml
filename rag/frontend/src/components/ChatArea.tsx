import { useState } from 'react';

const ChatArea: React.FC = () => {
  const [inputText, setInputText] = useState<string>('');
  const [responseText, setResponseText] = useState<string>('');

  const handleSubmit = async () => {
    if (!inputText.trim()) return;

    try {
      const response = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });
      if (!response.ok) throw new Error('Request failed');
      const data = await response.json();
      setResponseText(data.response || 'No result');
      setInputText('');
    } catch (error) {
      setResponseText('Error: ' + (error as Error).message);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleButtonClick = (e: React.FormEvent) => {
    e.preventDefault();
    handleSubmit();
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-y-auto border p-2 mb-2 bg-gray-50">
        <textarea
          className="w-full h-full resize-y border-none bg-transparent focus:outline-none"
          value={responseText}
          readOnly
          placeholder="Response will appear here..."
        />
      </div>
      <form onSubmit={handleButtonClick} className="flex">
        <textarea
          className="flex-1 border p-2 rounded-l resize-y"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          rows={2}
        />
        <button
          type="submit"
          className="bg-blue-500 text-white p-2 rounded-r hover:bg-blue-600"
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatArea;  
// return (
  //   <div className="h-full flex flex-col">
  //     <div className="flex-1 overflow-y-auto border p-2 mb-2 bg-gray-50">
  //       <p>Chat messages will appear here...</p>
  //     </div>
  //     <div className="flex">
  //       <input
  //         type="text"
  //         className="flex-1 border p-2 rounded-l"
  //         placeholder="Type a message..."
  //       />
  //       <button className="bg-blue-500 text-white p-2 rounded ml-2">Send</button>
  //     </div>
  //   </div>
  // );
// };

//export default ChatArea;
