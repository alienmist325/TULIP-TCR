## Descriptions of the changes made to the input

The first number is a grouping, and the second number is a specific test/ input within that?

Name | Description
-- | -- 
1 | Testing whether the model works on incomplete data
2 | Shuffled data
3 | Testing incomplete data, with the knowledge "\<MIS\>" is the correct token.



Name | Description | Output
-- | -- | --
1.1 | Removing the B column | Did not work.
1.2 | Add "\<UNK\>" to one of the entries. | Difficult to say whether it had an effect.
1.3 | Remove the MHC column | Did not work.
1.4 | Add the token to all entries in $\alpha$ | Again, unsure of the effect.
1.5 | Add an arbitrary "\<UNK1\>" | Did not cause an error, which is suspicious.
1.6 | Made the A column blank (but not removing) | I believe it caused an error, but I don't recall.
2.1 | Shuffled original data
2.2.1 | All the positives extracted
2.2.2 | All the positives, with CDR3a shuffled
3.1 | Putting in "\<MIS\>" as the CDR3b
3.2 | Replacting "\<MIS\>" with something arbitrary almost everywhere: "t"
3.3 | Full "\<MIS\>" in 2 columns (CDR3b and MHC)
3.4 | Replace 3.3's "\<MIS\>" with "t"
4 | Preparing to use the real data, so we have removed the binding column (this shouldn't be used). This doesn't work; binding column is needed
4.1 | Does the binding column have an effect on the output? Ran the test with all 1's and this was identical to a mix, so we conclude (thankfully) no; we can just set binding to 1
5 | The final, real data
6 | A new dataset