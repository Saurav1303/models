USE [wizerdevdb]
GO
/****** Object:  StoredProcedure [dbo].[SearchUsersByKeywords]    Script Date: 12-07-2024 21:40:52 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
ALTER PROCEDURE [dbo].[SearchUsersByKeywords]
    @Keywords NVARCHAR(MAX),
    @Limit INT
AS
BEGIN
    -- Declare a table variable to hold the matching user IDs
    DECLARE @MatchingUsers TABLE (UserId NVARCHAR(450));

    -- Split the keywords into individual terms
    DECLARE @Keyword NVARCHAR(256);
    DECLARE @KeywordIndex INT = 1;

    -- Split the keywords by spaces
    DECLARE @KeywordsTable TABLE (Keyword NVARCHAR(256));
    WHILE LEN(@Keywords) > 0
    BEGIN
        SET @KeywordIndex = CHARINDEX(' ', @Keywords);
        IF @KeywordIndex > 0
        BEGIN
            SET @Keyword = LEFT(@Keywords, @KeywordIndex - 1);
            SET @Keywords = RIGHT(@Keywords, LEN(@Keywords) - @KeywordIndex);
        END
        ELSE
        BEGIN
            SET @Keyword = @Keywords;
            SET @Keywords = '';
        END
        INSERT INTO @KeywordsTable (Keyword) VALUES (@Keyword);
    END

    -- Search through the tables and collect matching user IDs
    INSERT INTO @MatchingUsers (UserId)
    SELECT DISTINCT gd.UserId
    FROM [Gig].[GigDetails] gd
    INNER JOIN [Gig].[GigTags] gt ON gd.Id = gt.GigId
    INNER JOIN [Master].[SubCategory] sc ON gd.SubCategoryID = sc.Id
    INNER JOIN [Master].[Category] c ON gd.CategoryID = c.Id
    INNER JOIN [Identity].[UserSubCategory] usc ON gd.UserId = usc.UserId
    INNER JOIN [Identity].[UserSkills] us ON gd.UserId = us.UserId
    WHERE (
        EXISTS (SELECT 1 FROM @KeywordsTable k WHERE LOWER(gd.Description) LIKE '%' + LOWER(k.Keyword) + '%')
        OR EXISTS (SELECT 1 FROM @KeywordsTable k WHERE LOWER(gt.TagName) LIKE '%' + LOWER(k.Keyword) + '%')
        OR EXISTS (SELECT 1 FROM @KeywordsTable k WHERE LOWER(sc.Name) LIKE '%' + LOWER(k.Keyword) + '%')
        OR EXISTS (SELECT 1 FROM @KeywordsTable k WHERE LOWER(c.Name) LIKE '%' + LOWER(k.Keyword) + '%')
        OR EXISTS (SELECT 1 FROM @KeywordsTable k WHERE LOWER(us.Name) LIKE '%' + LOWER(k.Keyword) + '%')
        OR EXISTS (SELECT 1 FROM @KeywordsTable k WHERE DIFFERENCE(gd.Description, k.Keyword) >= 3)
        OR EXISTS (SELECT 1 FROM @KeywordsTable k WHERE DIFFERENCE(gt.TagName, k.Keyword) >= 3)
        OR EXISTS (SELECT 1 FROM @KeywordsTable k WHERE DIFFERENCE(sc.Name, k.Keyword) >= 3)
        OR EXISTS (SELECT 1 FROM @KeywordsTable k WHERE DIFFERENCE(c.Name, k.Keyword) >= 3)
        OR EXISTS (SELECT 1 FROM @KeywordsTable k WHERE DIFFERENCE(us.Name, k.Keyword) >= 3)
    );

    -- Return the list of matching users with a limit
    SELECT TOP (@Limit) UserId
    FROM @MatchingUsers;
END
