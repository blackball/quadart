#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdlib.h>

/*
----------------------------------------
Use quad to split render and split the image.

0. Input an image, and the iteration number <I, N>
1. Create root of the quad tree by using the I as the single element.
2. While i < N:
3.   Find the most detailed rectangle in the quad tree, and split it. Here we may need to split multiple nodes.
4.   i = i + 1
5. Render the quad tree to a image, by using the mean color of every rectangle.

@TODO: Split multiple nodes at once.
----------------------------------------
*/

struct color
{
        unsigned char r, g, b;
};

struct quadscore
{
        double var;
};

struct quadnode
{
        CvRect rect;
        struct quadscore score;
        struct quadnode *children[4];
};

static struct quadscore
empty_score()
{
        struct quadscore score;
        score.var = 0.; 
        return score;
}

struct quadnode *
quadnode_new(CvRect rect)
{
        struct quadnode *node = malloc(sizeof(*node));
        node->rect = rect;
        node->score = empty_score();
        for (int i = 0; i < 4; i++)
        {
                node->children[i] = NULL;
        }
        return node;
}

void
quadnode_free(struct quadnode *root)
{
        if (root)
        {
                for (int i = 0; i < 4; ++i)
                        quadnode_free(root->children[i]);
                free(root);
        }
}

/*
----------------------------------------
0 1
2 3
----------------------------------------
*/
static CvRect
sub_rect(const CvRect rect, int i)
{
        const int lw = rect.width >> 1;
        const int lh = rect.height >> 1;
        CvRect r;
        switch (i)
        {
        case 3:
                r.x = rect.x + lw;
                r.y = rect.y + lh;
                r.width = rect.width - lw;
                r.height = rect.height - lh;
                break;
        case 2:
                r.x = rect.x;
                r.y = rect.y + lh;
                r.width = lw;
                r.height = rect.height - lh;
                break;
        case 1:
                r.x = rect.x + lw;
                r.y = rect.y;
                r.width = rect.width - lw;
                r.height = lh;
                break;
        case 0:
                r.x = rect.x;
                r.y = rect.y;
                r.width = lw;
                r.height = lh;
                break;
        }
        return r;
}

static int
quadnode_isleaf(const struct quadnode *node)
{
        return  !(node->children[0] || node->children[1] ||
                  node->children[1] || node->children[3]);
}

/*
----------------------------------------
Split the 'node' into for
----------------------------------------
*/
int
quadnode_split(struct quadnode *node)
{
        if (quadnode_isleaf(node))
        {
                for (int i = 0; i < 4; i++)
                {
                        node->children[i] = quadnode_new(sub_rect(node->rect, i));
                }
        }
        return 0;
}

static double
maxd(double a, double b)
{
        return a > b ? a : b;
}

static double
variance(const IplImage *img, const CvRect rect)
{
        double meanA = 0., meanB = 0., meanC = 0.;
        double varA = 0., varB = 0., varC = 0.;

        for (int y = rect.y; y < rect.y + rect.height; ++y)
        {
                const unsigned char *row = (const unsigned char *)(img->imageData + y * img->widthStep);
                for (int x = rect.x; x < rect.x + rect.width; ++x)
                {
                        meanA += row[3 * x];
                        meanB += row[3 * x + 1];
                        meanC += row[3 * x + 2];
                }
        }

        const double n = rect.width * rect.height;
        meanA /= n;
        meanB /= n;
        meanC /= n;

        for (int y = rect.y; y < rect.y + rect.height; ++y)
        {
                const unsigned char *row = (const unsigned char *)(img->imageData + y * img->widthStep);
                for (int x = rect.x; x < rect.x + rect.width; ++x)
                {
                        varA += (row[3 * x] - meanA) * (row[3 * x] - meanA);
                        varB += (row[3 * x + 1] - meanB) * (row[3 * x + 1] - meanB);
                        varC += (row[3 * x + 2] - meanC) * (row[3 * x + 2] - meanC);
                }
        }
                
        return maxd(varA, maxd(varB, varC)) / n;
}

static struct quadscore
node_score(const IplImage *img, const CvRect rect)
{
        struct quadscore score;
        score.var = variance(img, rect);
        return score;
}

static int
score_lt(const struct quadscore a, const struct quadscore b)
{
        return a.var < b.var;
}

static int
rect_size(const CvRect rect)
{
        return rect.width * rect.height;
}

/*
----------------------------------------
  Find biggest score and its node in leaf node
----------------------------------------
*/
void
find_biggest_dfs(struct quadnode *root, struct quadnode **maxnode, struct quadscore *maxscore)
{
        if (!root) return;

        if (quadnode_isleaf(root) && rect_size(root->rect) > 16)
        {
                if (score_lt(*maxscore, root->score))
                {
                        *maxscore = root->score;
                        *maxnode = root;
                }
        }
        else
        {
                for (int i = 0; i < 4; i++)
                {
                        find_biggest_dfs(root->children[i], maxnode, maxscore);
                }
        }
}

static struct quadnode *
find_biggest(struct quadnode *root)
{
        struct quadnode *node = NULL;
        if (root)
        {
                struct quadscore score = empty_score();
                find_biggest_dfs(root, &node, &score);
        }
        return node;
}

/*
----------------------------------------
Find the biggest score, and split it.
Calculate new scores for new children.
----------------------------------------
*/
int
split_once(struct quadnode *root, const IplImage *img)
{
        struct quadnode *node = find_biggest(root);
        if (!node)
        {
                return -1;
        }
        quadnode_split(node);
        for (int i = 0; i < 4; ++i)
        {
                node->children[i]->score = node_score(img, node->children[i]->rect);
        }
        return 0;
}

static struct color
mean_color(const IplImage *img, const CvRect rect)
{
        int r = 0, g = 0, b = 0;
        for (int y = rect.y; y < rect.y + rect.height; y++)
        {
                const unsigned char *row = img->imageData + y * img->widthStep;
                for (int x = rect.x; x < rect.x + rect.width; x++)
                {
                        b += row[x * 3];
                        g += row[x * 3 + 1];
                        r += row[x * 3 + 2];
                }
        }

        const int n = rect.width *rect.height;
        struct color c;
        c.r = r / n;
        c.g = g / n;
        c.b = b / n;
        return c;
}

static void
set_color(IplImage *img, const CvRect roi, struct color c)
{
        /* Leave some spaces for drawing lines */
        CvRect r = roi;
        r.x += 1;
        r.y += 1;
        r.width -= 1;
        r.height -= 1;
        
        cvSetImageROI(img, r);
        cvSet(img, cvScalar(c.b, c.g, c.r, 0), 0);
        cvResetImageROI(img);
}

/*
----------------------------------------
Draw using mean color.

For the leaf node in the quad tree, use mean color to set the image.
----------------------------------------
*/
void
render(const struct quadnode *root, const IplImage *src, IplImage *out)
{
        if (root)
        {
                if (quadnode_isleaf(root))
                {
                        set_color(out, root->rect, mean_color(src, root->rect));
                }
                else
                {                        
                        const CvRect r = root->rect;
                        cvLine(out, cvPoint(r.x + r.width / 2, r.y), cvPoint(r.x + r.width / 2, r.y + r.height), cvScalar(0, 0, 0, 0), 1, 8, 0);
                        cvLine(out, cvPoint(r.x, r.y + r.height/2), cvPoint(r.x + r.width, r.y + r.height/2), cvScalar(0, 0, 0, 0), 1, 8, 0);
                        
                        for (int i = 0; i < 4; ++i)
                        {
                                render(root->children[i], src, out);
                        }
                }
        }
}

static CvRect
get_rect(const IplImage *img)
{
        CvRect rect = { 0, 0, img->width, img->height };
        return rect;
}

int
main(int argc, char *argv[])
{
        const int iteration_times = 1000;
        IplImage *img = cvLoadImage("me.png", 1);
        struct quadnode *root = quadnode_new(get_rect(img));

        /* Initialize the first score, In fact here we can
           give any score to root->score which is bigger than
           the score in empty_score();
        */
        root->score = node_score(img, root->rect);

        IplImage *out = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        
        cvNamedWindow("iteration", 1);
        for (int i = 0; i < iteration_times; i++)
        {
                if (-1 == split_once(root, img))
                        break;
                render(root, img, out);
                cvShowImage("iteration", out);
                cvWaitKey(10);
        }
        cvDestroyWindow("iteration");
        
        cvSaveImage("render.png", out, 0);
        cvReleaseImage(&out);
        cvReleaseImage(&img);
        quadnode_free(root);
        return 0;
}
