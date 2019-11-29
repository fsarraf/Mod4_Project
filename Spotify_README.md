
Spotify-  Executive Summary

Overview

What makes a hit song? To answer this question, we have analysed the key ingredients that play a role in making a song popular using data from Spotify over 20 years from 1999 to 2019. The objective is to predict the popularity of a song and reveal other useful insights. 

Method

Using the Spotify API, we randomly selected 200,000 songs between 1999 and 2019. For each song, we used composition information such as acousticness, energy, valence, danceability etc. (as defined by Spotify) to see which ones had an impact on popularity. In doing so, we segmented songs by genre, year and even sub-genre to identify patterns and check if these factors had any sort of impact on popularity. 

Target Audience

Our audience is bluechip record companies and independent music producers, who scout for new artists and are always looking out for the latest music trends to decide where to put their money next.

Findings

1. Twice as many pop songs were made over rock during this time period 
2. EDM and Pop emerged as the most popular genres
3. Jazz and Folk remain the least popular genres
4. While it is unlikely that an unpopular artist will ever create a popular song, being a popular artist also does not guarantee a hit song 
5. Only Rap had a moderate correlation with popularity( R2 = 0.4), with other genres being much less (Pop, Rock = 0.2)
6. Further segmenting songs by year lowered the predicitive power even more thus making it incapable as an indicator
7. Danceability remained more or less constant till 2011, which is the year it dropped sharply, and has been on an increasing trend ever since until 2019 when it dropped again
8. Post 2010, songs are becoming more acoustic, more dark and less energetic 

Conclusion and Recommendations

1. For record companies, a useful insight is that having a popular artist, being louder, more acoustic and being shorter in duration, actually makes a rap song more popular
2. We cannot exactly break down the ingredients of a hit song, as popularity is not strongly dependent on characteristics of a song
3. However, what may prove useful is market effect measures such as number of mentions / hashtags on social media when trying to predict the popularity of a song. This can also be used as a further area of study

Reference documents

1. Presentation - What makes a hit song _.pdf
2. Master jupyter notebook - Spotify_songs.ipynb
