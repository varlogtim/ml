interface Reference {
  id: number;
  title: string;
  author: string;
}

const references: Reference[] = [
  { id: 1, title: "Sample Ref 1", author: "Author A" },
  { id: 2, title: "Sample Ref 2", author: "Author B" },
];

const ReferenceTable: React.FC = () => {
  return (
    <table className="w-full border-collapse">
      <thead>
        <tr className="bg-gray-200">
          <th className="border p-2">ID</th>
          <th className="border p-2">Title</th>
          <th className="border p-2">Author</th>
        </tr>
      </thead>
      <tbody>
        {references.map((ref) => (
          <tr key={ref.id} className="hover:bg-gray-100">
            <td className="border p-2">{ref.id}</td>
            <td className="border p-2">{ref.title}</td>
            <td className="border p-2">{ref.author}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default ReferenceTable;
