// import { useState } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'

import { useState } from 'react';
import Navigation from './components/Navigation';
import AdminArea from './components/AdminArea';
////import ReferenceTable from './components/ReferenceTable';
import ChatArea from './components/ChatArea';


const App: React.FC = () => {
  const [selectedContent, setSelectedContent] = useState<string>('query');

  return (
    <div className="flex h-screen w-screen bg-gray-100">
      <div className="w-1/5 bg-gray-800 text-white">
        <Navigation onSelect={setSelectedContent} />
      </div>
      <div className="w-4/5 flex flex-col p-4">

        {/* Dynamic Content */}
        <div className="bg-white w-full flex-col flex rounded-lg shadow p-4">
          <h2 className="text-2xl font-bold mb-4 p-3">
            {selectedContent === 'query' ? 'Query' : 'Admin Area'}
          </h2>
          {selectedContent === 'query' ? <ChatArea /> : <AdminArea />}
        </div>
      </div>
    </div>
  )
}

export default App;
