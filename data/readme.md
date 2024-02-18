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
3.1 | Putting in "\<MIS\>" as the CDR3a
3.2 | Replacting "\<MIS\>" with something arbitrary almost everywhere: "t"