using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Autocomplete
{
    public class TrieNode
    {
        public char Value { get; set; }
        public Dictionary<char, TrieNode> Children { get; set; }
        public bool IsEndOfWord { get; set; }

        public TrieNode(char value)
        {
            Value = value;
            Children = new Dictionary<char, TrieNode>();
        }
    }

    public class Trie
    {
        private readonly TrieNode _root;

        public Trie()
        {
            _root = new TrieNode('\0');
        }

        public void AddWord(string word)
        {
            var node = _root;
            foreach (var c in word)
            {
                if (!node.Children.ContainsKey(c))
                {
                    node.Children[c] = new TrieNode(c);
                }
                node = node.Children[c];
            }
            node.IsEndOfWord = true;
        }

        public IEnumerable<string> GetSuggestions(string prefix)
        {
            var node = _root;
            foreach (var c in prefix)
            {
                if (!node.Children.ContainsKey(c))
                {
                    return Enumerable.Empty<string>();
                }
                node = node.Children[c];
            }

            return GetWords(node, prefix);
        }

        private IEnumerable<string> GetWords(TrieNode node, string prefix)
        {
            if (node.IsEndOfWord)
            {
                yield return prefix;
            }

            foreach (var child in node.Children)
            {
                yield return string.Join("", GetWords(child.Value, $"{prefix}{child.Key}"));
            }
        }
    }
}


public class TextFileReader
{
    //public List<string> ReadAndMergeKeywords(string filePath)
    public List<string> ReadAndMergeKeywords(string filePath)
    {
        //var mergedKeywords = new List<string>();
        var setKeywords = new HashSet<string>();
        try
        {
            // Read all lines from the file
            var lines = File.ReadAllLines(filePath);

            // Process each line
            foreach (var line in lines)
            {
                // Split the line into keywords and add them to the merged list
                //var keywords = line.Split(new[] { ' ', ',', ';', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                //mergedKeywords.AddRange(keywords);
                setKeywords.Add(line);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }

        //return mergedKeywords;
        return setKeywords.ToList();
    }
}

// Example usage
class Program
{
    static void Main()
    {
        var filePath = "C:\\Users\\manis\\OneDrive\\Desktop\\autocomplete_keywords_test.txt"; // Replace with the actual path to your text file
        var textFileReader = new TextFileReader();
        var mergedKeywords = textFileReader.ReadAndMergeKeywords(filePath);
        var trie = new Autocomplete.Trie();
        //trie.AddWord("apple");
        //trie.AddWord("banana");
        //trie.AddWord("cherry");
        //trie.AddWord("apricot");
        //trie.AddWord("classic");
        //trie.AddWord("query");
        // Print merged keywords
        Console.WriteLine("adding words to trie");
        foreach (var keyword in mergedKeywords)
        {
            //Console.WriteLine(keyword);
            trie.AddWord(keyword);
        }
        Console.WriteLine("---completed---");
        var suggestions = trie.GetSuggestions("ecommerce");
        foreach (string str in suggestions) { 
            Console.WriteLine($"{str}");
        }
            //Console.WriteLine(string.Join(", ", suggestions)); // Convert IEnumerable to string
    }
}