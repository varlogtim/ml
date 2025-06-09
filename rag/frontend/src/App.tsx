// import { useState } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'


import Navigation from './components/Navigation';
import ReferenceTable from './components/ReferenceTable';
import ChatArea from './components/ChatArea';


function App() {
  //const [count, setCount] = useState(0)

  return (
    <div className="flex h-screen w-screen bg-gray-100">
      {/* Navigation Pane: 20% */}
      <div className="w-50 flex-none bg-gray-800 text-white">
        <Navigation />
      </div>
      {/* Main Content: 80% */}
      <div className="flex-1 flex-col h-full flex p-4">
        <div className="flex space-x-4 h-full">
          {/* Reference Table */}
          <div className="w-1/2 h-full bg-white rounded-lg shadow">
            <div className="bg-white p-4">
              <h2 className="text-2xl font-bold">References</h2>
              <ReferenceTable />
            </div>
          </div>
          {/* Chat Area */}
          <div className="w-1/2 bg-white rounded-lg shadow">
            <div className="bg-white p-4">
              <h2 className="text-2xl font-bold">Chat</h2>
              <ChatArea />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
