import './App.css';
import { v7 as uuidv7 } from 'uuid';
import { Chat } from './features/chat';


const models = [
  {
    id: 'llama32-3b',
    name: 'Meta LLama 3.2 3B Instruct (AWS Bedrock)',
    selected: true,
  },
  {
    id: 'llama32-1b',
    name: 'Meta LLama 3.2 1B Instruct (AWS Bedrock)',
    selected: false,
  },
  {
    id: 'gpt-4o',
    name: 'GPT-4o',
    selected: false,
  },
  {
    id: 'gpt-4o-mini',
    name: 'GPT-4o mini',
    selected: false,
  },
  {
    id: 'cohere',
    name: 'Cohere Command R+ v1 (AWS Bedrock)',
    selected: false,
  },
  {
    id: 'nova-micro',
    name: 'Amazon Nove Micro (AWS Bedrock)',
    selected: false,
  },
  {
    id: 'nova-lite',
    name: 'Amazon Nove Lite (AWS Bedrock)',
    selected: false,
  },
  {
    id: 'nova-pro',
    name: 'Amazon Nove Pro (AWS Bedrock)',
    selected: false,
  },
];

function App() {
  const sessionId = uuidv7();

  return (
    <Chat models={models} sessionId={sessionId} />
  );
}

export default App;
